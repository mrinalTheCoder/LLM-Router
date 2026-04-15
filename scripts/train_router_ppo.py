#!/usr/bin/env python3
"""Train an online LLM-router policy with PPO and full W&B logging.

This script is online-first: every action chooses a model, performs a live query, and
uses returned metrics (correctness, latency, energy, telemetry) for reward and logging.

Example:
python scripts/train_router_ppo.py \
  --train-limit 512 \
  --val-limit 128 \
  --total-timesteps 5000 \
  --learning-rate 3e-4 \
  --encoder-training-mode frozen \
  --reward-alpha 1.0 \
  --reward-beta 0.2 \
  --reward-gamma 0.2 \
  --model-pool "llama3.2:3b,qwen2.5:3b,granite3.3:8b" \
  --wandb-project llm-router \
  --wandb-mode online
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise RuntimeError(
        "Missing dependency: gymnasium. Install requirements before running training."
    ) from exc

try:
    import torch
except ImportError as exc:
    raise RuntimeError(
        "Missing dependency: torch. Install requirements before running training."
    ) from exc

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as exc:
    raise RuntimeError(
        "Missing dependency: stable-baselines3. Install requirements before running training."
    ) from exc

try:
    import wandb
except ImportError:
    wandb = None

try:
    import yaml
except ImportError:
    yaml = None

from benchmark_mmlu_csv import run_ollama_prompt

CHOICE_LABELS = ("A", "B", "C", "D")
ANSWER_RE = re.compile(r"\b([ABCD])\b")
ENCODER_MODE_CHOICES = ("frozen", "finetune")


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: int
    subject: str
    question: str
    choices: Tuple[str, str, str, str]
    truth: str

    @property
    def router_text(self) -> str:
        return "\n".join(
            [
                self.question,
                f"A. {self.choices[0]}",
                f"B. {self.choices[1]}",
                f"C. {self.choices[2]}",
                f"D. {self.choices[3]}",
            ]
        )


@dataclass(frozen=True)
class RewardConfig:
    alpha: float
    beta: float
    gamma: float
    latency_norm_s: float
    energy_norm_j: float

    def compute(
        self, correct: int, latency_s: Optional[float], energy_joules: Optional[float]
    ) -> Dict[str, float]:
        latency_norm = 1.0
        if latency_s is not None and self.latency_norm_s > 0:
            latency_norm = float(np.clip(latency_s / self.latency_norm_s, 0.0, 1.0))

        energy_norm = 1.0
        if energy_joules is not None and self.energy_norm_j > 0:
            energy_norm = float(np.clip(energy_joules / self.energy_norm_j, 0.0, 1.0))

        reward_correct = self.alpha * float(correct)
        reward_latency_penalty = self.beta * latency_norm
        reward_energy_penalty = self.gamma * energy_norm
        total_reward = reward_correct - reward_latency_penalty - reward_energy_penalty
        return {
            "reward_total": total_reward,
            "reward_correct": reward_correct,
            "reward_latency_penalty": reward_latency_penalty,
            "reward_energy_penalty": reward_energy_penalty,
            "latency_norm": latency_norm,
            "energy_norm": energy_norm,
        }


@dataclass
class ModelOutcome:
    model_name: str
    response_text: str
    predicted: Optional[str]
    truth: str
    correct: int
    latency_s: Optional[float]
    ttft_s: Optional[float]
    wall_tokens_per_s: Optional[float]
    eval_tokens_per_s: Optional[float]
    prompt_eval_tokens_per_s: Optional[float]
    energy_joules: Optional[float]
    energy_source: Optional[str]
    average_power_watts: Optional[float]
    cpu_percent: Optional[float]
    cpu_effort_cpu_seconds: Optional[float]
    rss_delta_bytes: Optional[float]
    gpu_utilization_avg_pct: Optional[float]
    gpu_vram_used_avg_mib: Optional[float]
    gpu_power_avg_watts: Optional[float]
    gpu_power_min_watts: Optional[float]
    gpu_power_max_watts: Optional[float]


class OutcomeProvider(Protocol):
    """Provider abstraction for model outcomes.

    Online provider is used now; a cache/database provider can implement this same
    interface later without changing the training loop.
    """

    def query(self, prompt: PromptRecord, model_name: str) -> ModelOutcome:
        ...


@dataclass(frozen=True)
class OnlineProviderConfig:
    host: str
    timeout_s: float
    gpu_layers: Optional[int]
    gpu_device: int
    power_sample_interval_s: float


class OnlineOllamaProvider:
    def __init__(self, config: OnlineProviderConfig):
        self.config = config

    def query(self, prompt: PromptRecord, model_name: str) -> ModelOutcome:
        query_prompt = build_prompt(prompt.question, prompt.choices)
        metrics = run_ollama_prompt(
            prompt=query_prompt,
            model=model_name,
            host=self.config.host,
            timeout_s=self.config.timeout_s,
            gpu_layers=self.config.gpu_layers,
            gpu_device=self.config.gpu_device,
            power_sample_interval_s=self.config.power_sample_interval_s,
        )

        predicted = extract_answer_letter(metrics["response_text"])
        correct = int(predicted == prompt.truth) if predicted is not None else 0
        return ModelOutcome(
            model_name=model_name,
            response_text=metrics["response_text"],
            predicted=predicted,
            truth=prompt.truth,
            correct=correct,
            latency_s=to_opt_float(metrics.get("total_latency_s")),
            ttft_s=to_opt_float(metrics.get("ttft_s")),
            wall_tokens_per_s=to_opt_float(metrics.get("wall_tokens_per_s")),
            eval_tokens_per_s=to_opt_float(metrics.get("eval_tokens_per_s")),
            prompt_eval_tokens_per_s=to_opt_float(metrics.get("prompt_eval_tokens_per_s")),
            energy_joules=to_opt_float(metrics.get("energy_joules")),
            energy_source=str(metrics.get("energy_source")) if metrics.get("energy_source") else None,
            average_power_watts=to_opt_float(metrics.get("average_power_watts")),
            cpu_percent=to_opt_float(metrics.get("cpu_percent")),
            cpu_effort_cpu_seconds=to_opt_float(metrics.get("cpu_effort_cpu_seconds")),
            rss_delta_bytes=to_opt_float(metrics.get("rss_delta_bytes")),
            gpu_utilization_avg_pct=to_opt_float(metrics.get("gpu_utilization_avg_pct")),
            gpu_vram_used_avg_mib=to_opt_float(metrics.get("gpu_vram_used_avg_mib")),
            gpu_power_avg_watts=to_opt_float(metrics.get("gpu_power_avg_watts")),
            gpu_power_min_watts=to_opt_float(metrics.get("gpu_power_min_watts")),
            gpu_power_max_watts=to_opt_float(metrics.get("gpu_power_max_watts")),
        )


RouterObservation = Union[np.ndarray, Dict[str, np.ndarray]]
ObservationData = Union[np.ndarray, Dict[str, np.ndarray]]


class ObservationStore:
    """Container that abstracts frozen vector obs vs tokenized dict obs."""

    def __init__(self, data: ObservationData):
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError(
                    "Embedding observations must be a 2D array of shape (N, D)."
                )
            self.kind = "embedding"
            self.data: ObservationData = data.astype(np.float32, copy=False)
            obs_dim = int(data.shape[1])
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self._zero_observation: RouterObservation = np.zeros(
                (obs_dim,), dtype=np.float32
            )
            return

        if not isinstance(data, dict):
            raise ValueError("ObservationStore expects either ndarray or dict observations.")

        input_ids = np.asarray(data.get("input_ids"), dtype=np.int64)
        attention_mask = np.asarray(data.get("attention_mask"), dtype=np.int64)
        if input_ids.ndim != 2:
            raise ValueError(
                "Token observations must provide input_ids with shape (N, T)."
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape.")

        self.kind = "tokenized"
        self.data = {"input_ids": input_ids, "attention_mask": attention_mask}
        seq_len = int(input_ids.shape[1])
        self.observation_space = spaces.Dict(
            {
                "input_ids": spaces.Box(
                    low=0,
                    high=np.iinfo(np.int32).max,
                    shape=(seq_len,),
                    dtype=np.int64,
                ),
                "attention_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(seq_len,),
                    dtype=np.int64,
                ),
            }
        )
        self._zero_observation = {
            "input_ids": np.zeros((seq_len,), dtype=np.int64),
            "attention_mask": np.zeros((seq_len,), dtype=np.int64),
        }

    def __len__(self) -> int:
        if self.kind == "embedding":
            return int(self.data.shape[0])  # type: ignore[union-attr]
        input_ids = self.data["input_ids"]  # type: ignore[index]
        return int(input_ids.shape[0])

    def get(self, idx: int) -> RouterObservation:
        if self.kind == "embedding":
            return np.asarray(self.data[idx], dtype=np.float32).copy()  # type: ignore[index]
        return {
            key: np.asarray(value[idx], dtype=np.int64).copy()
            for key, value in self.data.items()  # type: ignore[union-attr]
        }

    def zero(self) -> RouterObservation:
        if self.kind == "embedding":
            return np.asarray(self._zero_observation, dtype=np.float32).copy()
        return {
            key: np.asarray(value, dtype=np.int64).copy()
            for key, value in self._zero_observation.items()  # type: ignore[union-attr]
        }

    def subset(self, length: int) -> "ObservationStore":
        if self.kind == "embedding":
            return ObservationStore(self.data[:length])  # type: ignore[index]
        return ObservationStore(
            {
                key: value[:length]
                for key, value in self.data.items()  # type: ignore[union-attr]
            }
        )


class TransformerClsFeaturesExtractor(BaseFeaturesExtractor):
    """Shared encoder used by PPO when encoder_training_mode=finetune."""

    def __init__(self, observation_space: spaces.Dict, encoder_name: str):
        super().__init__(observation_space, features_dim=1)
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: transformers. Install requirements before running training."
            ) from exc

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                f"Encoder {encoder_name!r} does not expose config.hidden_size."
            )
        self._features_dim = int(hidden_size)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = observations["input_ids"].long()
        attention_mask = observations["attention_mask"].long()
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            raise RuntimeError("Transformer encoder did not return last_hidden_state.")
        return last_hidden_state[:, 0, :]


class RouterBanditEnv(gym.Env):
    """Single-step contextual bandit environment for model routing."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        records: Sequence[PromptRecord],
        observations: ObservationStore,
        model_pool: Sequence[str],
        provider: OutcomeProvider,
        reward_config: RewardConfig,
        seed: int,
    ) -> None:
        super().__init__()
        if len(records) == 0:
            raise ValueError("RouterBanditEnv requires at least one prompt record.")
        if len(observations) != len(records):
            raise ValueError("Observation row count must equal number of prompt records.")
        if len(model_pool) < 2:
            raise ValueError("Model pool must contain at least two candidate models.")

        self.records = list(records)
        self.observations = observations
        self.model_pool = list(model_pool)
        self.provider = provider
        self.reward_config = reward_config
        self._rng = np.random.default_rng(seed)
        self._current_index: Optional[int] = None
        self.observation_space = self.observations.observation_space
        self.action_space = spaces.Discrete(len(self.model_pool))

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[RouterObservation, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current_index = int(self._rng.integers(0, len(self.records)))
        record = self.records[self._current_index]
        info = {"prompt_id": record.prompt_id, "subject": record.subject}
        return self.observations.get(self._current_index), info

    def step(self, action: int):
        if self._current_index is None:
            raise RuntimeError("step() called before reset().")

        action_idx = int(action)
        if action_idx < 0 or action_idx >= len(self.model_pool):
            raise ValueError(f"Action index out of bounds: {action_idx}")

        record = self.records[self._current_index]
        selected_model = self.model_pool[action_idx]
        outcome = self.provider.query(record, selected_model)
        reward_parts = self.reward_config.compute(
            correct=outcome.correct,
            latency_s=outcome.latency_s,
            energy_joules=outcome.energy_joules,
        )

        info = {
            "prompt_id": record.prompt_id,
            "subject": record.subject,
            "model_name": selected_model,
            "action_index": action_idx,
            "truth": record.truth,
            "predicted": outcome.predicted,
            "correct": outcome.correct,
            "latency_s": outcome.latency_s,
            "ttft_s": outcome.ttft_s,
            "wall_tokens_per_s": outcome.wall_tokens_per_s,
            "eval_tokens_per_s": outcome.eval_tokens_per_s,
            "prompt_eval_tokens_per_s": outcome.prompt_eval_tokens_per_s,
            "energy_joules": outcome.energy_joules,
            "energy_source": outcome.energy_source,
            "average_power_watts": outcome.average_power_watts,
            "cpu_percent": outcome.cpu_percent,
            "cpu_effort_cpu_seconds": outcome.cpu_effort_cpu_seconds,
            "rss_delta_bytes": outcome.rss_delta_bytes,
            "gpu_utilization_avg_pct": outcome.gpu_utilization_avg_pct,
            "gpu_vram_used_avg_mib": outcome.gpu_vram_used_avg_mib,
            "gpu_power_avg_watts": outcome.gpu_power_avg_watts,
            "gpu_power_min_watts": outcome.gpu_power_min_watts,
            "gpu_power_max_watts": outcome.gpu_power_max_watts,
            **reward_parts,
        }

        self._current_index = None
        terminated = True
        truncated = False
        return self.observations.zero(), reward_parts["reward_total"], terminated, truncated, info


class RouterEvaluator:
    def __init__(
        self,
        records: Sequence[PromptRecord],
        observations: ObservationStore,
        model_pool: Sequence[str],
        provider: OutcomeProvider,
        reward_config: RewardConfig,
        limit: Optional[int] = None,
    ) -> None:
        self.records = list(records[:limit] if limit is not None else records)
        self.observations = observations.subset(len(self.records))
        self.model_pool = list(model_pool)
        self.provider = provider
        self.reward_config = reward_config

    def evaluate(self, model: PPO) -> Dict[str, float]:
        if len(self.records) == 0:
            return {}

        total_reward = 0.0
        total_correct = 0
        total_energy = 0.0
        total_energy_count = 0
        total_latency = 0.0
        total_latency_count = 0
        total_ttft = 0.0
        total_ttft_count = 0
        total_wall_tps = 0.0
        total_wall_tps_count = 0
        total_eval_tps = 0.0
        total_eval_tps_count = 0
        action_counter = Counter()
        model_correct_counter = Counter()
        model_total_counter = Counter()

        for idx, record in enumerate(self.records):
            obs = self.observations.get(idx)
            action, _ = model.predict(obs, deterministic=True)
            action_idx = int(action)
            selected_model = self.model_pool[action_idx]
            action_counter[selected_model] += 1
            outcome = self.provider.query(record, selected_model)
            model_total_counter[selected_model] += 1
            if outcome.correct:
                model_correct_counter[selected_model] += 1

            parts = self.reward_config.compute(
                correct=outcome.correct,
                latency_s=outcome.latency_s,
                energy_joules=outcome.energy_joules,
            )
            total_reward += parts["reward_total"]
            total_correct += outcome.correct

            if outcome.energy_joules is not None:
                total_energy += outcome.energy_joules
                total_energy_count += 1
            if outcome.latency_s is not None:
                total_latency += outcome.latency_s
                total_latency_count += 1
            if outcome.ttft_s is not None:
                total_ttft += outcome.ttft_s
                total_ttft_count += 1
            if outcome.wall_tokens_per_s is not None:
                total_wall_tps += outcome.wall_tokens_per_s
                total_wall_tps_count += 1
            if outcome.eval_tokens_per_s is not None:
                total_eval_tps += outcome.eval_tokens_per_s
                total_eval_tps_count += 1

        total = float(len(self.records))
        accuracy = total_correct / total if total > 0 else 0.0
        avg_energy = (total_energy / total_energy_count) if total_energy_count > 0 else None
        avg_latency = (total_latency / total_latency_count) if total_latency_count > 0 else None
        avg_ttft = (total_ttft / total_ttft_count) if total_ttft_count > 0 else None
        avg_wall_tps = (total_wall_tps / total_wall_tps_count) if total_wall_tps_count > 0 else None
        avg_eval_tps = (total_eval_tps / total_eval_tps_count) if total_eval_tps_count > 0 else None
        accuracy_per_joule = (accuracy / avg_energy) if avg_energy and avg_energy > 0 else None

        metrics: Dict[str, float] = {
            "eval/episodes": total,
            "eval/mean_reward": total_reward / total,
            "eval/accuracy": accuracy,
        }
        if avg_latency is not None:
            metrics["eval/avg_latency_s"] = avg_latency
        if avg_ttft is not None:
            metrics["eval/avg_ttft_s"] = avg_ttft
        if avg_energy is not None:
            metrics["eval/avg_energy_joules"] = avg_energy
        if accuracy_per_joule is not None:
            metrics["eval/accuracy_per_joule"] = accuracy_per_joule
        if avg_wall_tps is not None:
            metrics["eval/avg_wall_tokens_per_s"] = avg_wall_tps
        if avg_eval_tps is not None:
            metrics["eval/avg_eval_tokens_per_s"] = avg_eval_tps

        for model_name in self.model_pool:
            selected = action_counter.get(model_name, 0)
            metrics[f"eval/model_selection_rate/{model_name}"] = selected / total
            chosen_total = model_total_counter.get(model_name, 0)
            if chosen_total > 0:
                metrics[f"eval/model_accuracy/{model_name}"] = (
                    model_correct_counter.get(model_name, 0) / float(chosen_total)
                )
        return metrics


class WandbMetricsCallback(BaseCallback):
    def __init__(
        self,
        run: "wandb.sdk.wandb_run.Run",
        model_pool: Sequence[str],
        eval_runner: Optional[RouterEvaluator],
        output_dir: Path,
        log_every_steps: int,
        eval_every_steps: int,
        checkpoint_metric: str,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.run = run
        self.model_pool = list(model_pool)
        self.eval_runner = eval_runner
        self.output_dir = output_dir
        self.log_every_steps = max(log_every_steps, 1)
        self.eval_every_steps = max(eval_every_steps, 0)
        self.checkpoint_metric = checkpoint_metric
        self._buffer: Dict[str, List[float]] = defaultdict(list)
        self._action_counter: Counter = Counter()
        self._last_eval_step = 0
        self._best_metric: Optional[float] = None
        self._best_model_path = self.output_dir / "best_router_policy"

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            model_name = info.get("model_name")
            if isinstance(model_name, str):
                self._action_counter[model_name] += 1

            for key in (
                "reward_total",
                "reward_correct",
                "reward_latency_penalty",
                "reward_energy_penalty",
                "latency_norm",
                "energy_norm",
                "correct",
                "latency_s",
                "ttft_s",
                "energy_joules",
                "wall_tokens_per_s",
                "eval_tokens_per_s",
                "prompt_eval_tokens_per_s",
                "average_power_watts",
                "cpu_percent",
                "cpu_effort_cpu_seconds",
                "rss_delta_bytes",
                "gpu_utilization_avg_pct",
                "gpu_vram_used_avg_mib",
                "gpu_power_avg_watts",
                "gpu_power_min_watts",
                "gpu_power_max_watts",
            ):
                value = info.get(key)
                if value is None:
                    continue
                self._buffer[key].append(float(value))

        if self.num_timesteps % self.log_every_steps == 0:
            self._log_train_metrics()

        if (
            self.eval_runner is not None
            and self.eval_every_steps > 0
            and (self.num_timesteps - self._last_eval_step) >= self.eval_every_steps
        ):
            self._last_eval_step = self.num_timesteps
            eval_metrics = self.eval_runner.evaluate(self.model)
            if eval_metrics:
                eval_metrics["train/step"] = float(self.num_timesteps)
                self.run.log(eval_metrics)
                self._maybe_save_best(eval_metrics)
        return True

    def _log_train_metrics(self) -> None:
        payload: Dict[str, float] = {"train/step": float(self.num_timesteps)}
        for key, values in self._buffer.items():
            if not values:
                continue
            payload[f"train/{key}"] = float(np.mean(values))
        self._buffer.clear()

        total_actions = sum(self._action_counter.values())
        if total_actions > 0:
            for model_name in self.model_pool:
                payload[f"train/model_selection_rate/{model_name}"] = (
                    self._action_counter.get(model_name, 0) / float(total_actions)
                )

        logger_values = getattr(self.model.logger, "name_to_value", {})
        for key, value in logger_values.items():
            if isinstance(value, (float, int, np.floating, np.integer)):
                payload[f"sb3/{key}"] = float(value)

        self.run.log(payload)

    def _maybe_save_best(self, eval_metrics: Dict[str, float]) -> None:
        metric_value = eval_metrics.get(self.checkpoint_metric)
        if metric_value is None:
            return
        if self._best_metric is None or metric_value > self._best_metric:
            self._best_metric = metric_value
            self.model.save(str(self._best_model_path))
            artifact = wandb.Artifact("best-router-policy", type="model")
            artifact.add_file(str(self._best_model_path) + ".zip")
            self.run.log_artifact(artifact)


def to_opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_prompt(question: str, choices: Sequence[str]) -> str:
    joined_choices = "\n".join(
        f"{label}. {choice}" for label, choice in zip(CHOICE_LABELS, choices)
    )
    return (
        "Answer the following multiple-choice question.\n"
        "Return only the single best option letter: A, B, C, or D.\n\n"
        f"Question: {question}\n"
        f"{joined_choices}\n\n"
        "Answer:"
    )


def extract_answer_letter(text: str) -> Optional[str]:
    matches = ANSWER_RE.findall(text.upper())
    if matches:
        return matches[0]
    stripped = text.strip().upper()
    if stripped and stripped[0] in CHOICE_LABELS:
        return stripped[0]
    return None


def parse_truth(answer: Any) -> str:
    if isinstance(answer, int):
        return CHOICE_LABELS[answer]
    answer_text = str(answer).strip().upper()
    if answer_text in CHOICE_LABELS:
        return answer_text
    if answer_text.isdigit() and int(answer_text) in range(4):
        return CHOICE_LABELS[int(answer_text)]
    raise ValueError(f"Unsupported answer format: {answer!r}")


def load_mmlu_records(
    dataset_name: str, dataset_config: str, split: str, limit: Optional[int]
) -> List[PromptRecord]:
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: polars. Install requirements before running training."
        ) from exc

    parquet_path = f"hf://datasets/{dataset_name}/{dataset_config}/{split}-*.parquet"
    frame = pl.read_parquet(parquet_path)
    if limit is not None:
        frame = frame.head(limit)

    records: List[PromptRecord] = []
    for idx, row in enumerate(frame.to_dicts()):
        choices = row.get("choices")
        if not isinstance(choices, (list, tuple)) or len(choices) < 4:
            raise ValueError(f"Unexpected choices format at index {idx}: {choices!r}")
        records.append(
            PromptRecord(
                prompt_id=idx,
                subject=str(row.get("subject", "unknown")),
                question=str(row.get("question", "")),
                choices=(
                    str(choices[0]),
                    str(choices[1]),
                    str(choices[2]),
                    str(choices[3]),
                ),
                truth=parse_truth(row.get("answer")),
            )
        )
    return records


def compute_embeddings(
    records: Sequence[PromptRecord],
    encoder_name: str,
    encoder_device: str,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. Install requirements before running training."
        ) from exc

    encoder = SentenceTransformer(encoder_name, device=encoder_device)
    texts = [record.router_text for record in records]
    vectors = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def load_transformers_tokenizer(encoder_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: transformers. Install requirements before running training."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
    if tokenizer.pad_token is None:
        fallback_pad_token = tokenizer.eos_token or tokenizer.unk_token
        if fallback_pad_token is None:
            raise ValueError(
                f"Encoder tokenizer {encoder_name!r} has no pad, eos, or unk token."
            )
        tokenizer.pad_token = fallback_pad_token
    return tokenizer


def compute_tokenized_observations(
    records: Sequence[PromptRecord],
    tokenizer: Any,
    max_length: int,
) -> ObservationStore:
    texts = [record.router_text for record in records]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="np",
    )
    input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        if tokenizer.pad_token_id is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
        else:
            attention_mask = (input_ids != int(tokenizer.pad_token_id)).astype(np.int64)
    return ObservationStore(
        {
            "input_ids": input_ids,
            "attention_mask": np.asarray(attention_mask, dtype=np.int64),
        }
    )


def parse_model_pool(raw: str) -> List[str]:
    if isinstance(raw, str):
        models = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple)):
        models = [str(item).strip() for item in raw if str(item).strip()]
    else:
        raise ValueError(f"Unsupported model-pool value: {raw!r}")
    if not models:
        raise ValueError("Model pool cannot be empty.")
    return models


def parse_hidden_sizes(raw: Any, *, arg_name: str) -> List[int]:
    if isinstance(raw, str):
        sizes = [int(item.strip()) for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple)):
        sizes = [int(item) for item in raw]
    else:
        raise ValueError(f"Unsupported {arg_name} value: {raw!r}")
    if not sizes:
        raise ValueError(f"At least one hidden layer size is required for {arg_name}.")
    return sizes


def validate_args(args: argparse.Namespace) -> None:
    if args.encoder_training_mode not in ENCODER_MODE_CHOICES:
        raise ValueError(
            "encoder_training_mode must be specified for every run and be one of: "
            + ", ".join(ENCODER_MODE_CHOICES)
        )
    if args.encoder_max_length <= 0:
        raise ValueError("encoder_max_length must be a positive integer.")
    if args.encoder_training_mode == "finetune" and args.embedding_normalize:
        raise ValueError(
            "--embedding-normalize is only supported with encoder-training-mode=frozen."
        )


def build_observation_store(
    records: Sequence[PromptRecord], args: argparse.Namespace
) -> ObservationStore:
    if args.encoder_training_mode == "frozen":
        return ObservationStore(
            compute_embeddings(
                records,
                encoder_name=args.encoder_name,
                encoder_device=args.encoder_device,
                batch_size=args.embedding_batch_size,
                normalize=args.embedding_normalize,
            )
        )

    tokenizer = load_transformers_tokenizer(args.encoder_name)
    return compute_tokenized_observations(
        records,
        tokenizer=tokenizer,
        max_length=args.encoder_max_length,
    )


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_yaml_mapping(
    obj: Dict[str, Any], prefix: str = ""
) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in obj.items():
        key_norm = str(key).strip().replace("-", "_")
        merged_key = f"{prefix}_{key_norm}" if prefix else key_norm
        if isinstance(value, dict):
            flat.update(_flatten_yaml_mapping(value, prefix=merged_key))
        else:
            flat[merged_key] = value
    return flat


def _load_yaml_defaults(
    yaml_path: str, valid_keys: Sequence[str]
) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. Install requirements first."
        )
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("YAML config must be a mapping/object at the top level.")

    flattened = _flatten_yaml_mapping(payload)
    valid = set(valid_keys)
    defaults: Dict[str, Any] = {}
    unknown = []
    for key, value in flattened.items():
        if key in valid:
            defaults[key] = value
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(
            "Unknown YAML config keys: "
            + ", ".join(sorted(unknown))
            + ". Use argument destination names (e.g., learning_rate, reward_alpha)."
        )
    return defaults


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an online PPO policy for LLM routing with reward = "
            "alpha*correct - beta*latency - gamma*energy and full W&B logging."
        )
    )
    parser.add_argument("--dataset", default="cais/mmlu", help="HuggingFace dataset ID.")
    parser.add_argument("--config", default="all", help="Dataset config.")
    parser.add_argument(
        "--config-yaml",
        default=None,
        help=(
            "Optional YAML config file. Keys match argument destinations "
            "(e.g., learning_rate, reward_alpha). CLI flags override YAML values."
        ),
    )
    parser.add_argument("--train-split", default="test", help="Dataset split for training.")
    parser.add_argument(
        "--val-split", default="validation", help="Dataset split for periodic evaluation."
    )
    parser.add_argument("--train-limit", type=int, default=512, help="Training prompt cap.")
    parser.add_argument("--val-limit", type=int, default=128, help="Validation prompt cap.")

    parser.add_argument(
        "--model-pool",
        default=(
            "llama3-chatqa:8b,llama3.2:3b,qwen2.5:3b,dolphin3:8b,granite3.3:8b,"
            "mathstral:7b,meditron:7b,sailor2:8b"
        ),
        help="Comma-separated model names available to the router.",
    )
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout (s).")
    parser.add_argument("--gpu-layers", type=int, default=999, help="Ollama num_gpu option.")
    parser.add_argument("--gpu-device", type=int, default=0, help="GPU device index.")
    parser.add_argument(
        "--power-sample-interval",
        type=float,
        default=0.1,
        help="Seconds between nvidia-smi telemetry samples.",
    )

    parser.add_argument("--reward-alpha", type=float, default=1.0, help="Correctness weight.")
    parser.add_argument("--reward-beta", type=float, default=0.2, help="Latency penalty weight.")
    parser.add_argument("--reward-gamma", type=float, default=0.2, help="Energy penalty weight.")
    parser.add_argument(
        "--latency-norm-s",
        type=float,
        default=3.0,
        help="Latency normalization denominator in seconds.",
    )
    parser.add_argument(
        "--energy-norm-j",
        type=float,
        default=200.0,
        help="Energy normalization denominator in joules.",
    )

    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--total-timesteps", type=int, default=5000, help="Total PPO timesteps.")
    parser.add_argument("--n-steps", type=int, default=256, help="Rollout steps per update.")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size.")
    parser.add_argument("--ppo-epochs", type=int, default=10, help="PPO epochs per update.")
    parser.add_argument("--gamma-rl", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient.")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max grad norm.")
    parser.add_argument(
        "--actor-hidden-sizes",
        default="256,128",
        help="Comma-separated hidden sizes for the actor head MLP.",
    )
    parser.add_argument(
        "--critic-hidden-sizes",
        default="256,64",
        help="Comma-separated hidden sizes for the critic head MLP.",
    )
    parser.add_argument("--n-envs", type=int, default=1, help="Number of vectorized envs.")
    parser.add_argument("--ppo-device", default="auto", help="SB3 torch device.")
    parser.add_argument(
        "--encoder-training-mode",
        choices=ENCODER_MODE_CHOICES,
        default=None,
        help="Required per run: use frozen precomputed embeddings or end-to-end finetuning.",
    )

    parser.add_argument(
        "--encoder-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Encoder checkpoint for frozen embeddings or fine-tunable transformer features.",
    )
    parser.add_argument(
        "--encoder-device",
        default="cpu",
        help="Encoder device for frozen embedding precomputation.",
    )
    parser.add_argument(
        "--encoder-max-length",
        type=int,
        default=256,
        help="Maximum token length used when encoder-training-mode=finetune.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Embedding batch size for frozen prompt encoding.",
    )
    parser.add_argument(
        "--embedding-normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize sentence embeddings before PPO in frozen mode.",
    )

    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=1000,
        help="Run deterministic evaluation every N training timesteps (0 disables).",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=100,
        help="W&B train metric logging interval in timesteps.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        default="eval/accuracy_per_joule",
        help="Evaluation metric used for best-checkpoint selection.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--output-dir", default="outputs/router_ppo", help="Output directory.")

    parser.add_argument("--wandb-project", default="llm-router", help="W&B project.")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity/team.")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name.")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="W&B logging mode.",
    )
    return parser


def parse_args_with_yaml(parser: argparse.ArgumentParser) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config-yaml", default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()
    if bootstrap_args.config_yaml:
        valid_keys = [action.dest for action in parser._actions]
        yaml_defaults = _load_yaml_defaults(bootstrap_args.config_yaml, valid_keys)
        parser.set_defaults(**yaml_defaults)
    return parser.parse_args()


def main() -> None:
    parser = build_arg_parser()
    args = parse_args_with_yaml(parser)
    validate_args(args)
    set_global_seeds(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_pool = parse_model_pool(args.model_pool)
    actor_hidden_sizes = parse_hidden_sizes(
        args.actor_hidden_sizes, arg_name="actor_hidden_sizes"
    )
    critic_hidden_sizes = parse_hidden_sizes(
        args.critic_hidden_sizes, arg_name="critic_hidden_sizes"
    )

    train_records = load_mmlu_records(
        dataset_name=args.dataset,
        dataset_config=args.config,
        split=args.train_split,
        limit=args.train_limit,
    )
    val_records = load_mmlu_records(
        dataset_name=args.dataset,
        dataset_config=args.config,
        split=args.val_split,
        limit=args.val_limit,
    )
    if len(train_records) == 0:
        raise RuntimeError("No training records loaded.")

    train_observations = build_observation_store(train_records, args)
    val_observations = build_observation_store(val_records, args)

    reward_config = RewardConfig(
        alpha=args.reward_alpha,
        beta=args.reward_beta,
        gamma=args.reward_gamma,
        latency_norm_s=args.latency_norm_s,
        energy_norm_j=args.energy_norm_j,
    )
    provider_config = OnlineProviderConfig(
        host=args.host,
        timeout_s=args.timeout,
        gpu_layers=args.gpu_layers,
        gpu_device=args.gpu_device,
        power_sample_interval_s=args.power_sample_interval,
    )

    def make_env(env_seed: int):
        def _factory():
            provider = OnlineOllamaProvider(provider_config)
            return RouterBanditEnv(
                records=train_records,
                observations=train_observations,
                model_pool=model_pool,
                provider=provider,
                reward_config=reward_config,
                seed=env_seed,
            )

        return _factory

    env = DummyVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])

    policy_kwargs = {"net_arch": {"pi": actor_hidden_sizes, "vf": critic_hidden_sizes}}
    policy_name = "MlpPolicy"
    if args.encoder_training_mode == "finetune":
        policy_name = "MultiInputPolicy"
        policy_kwargs.update(
            {
                "features_extractor_class": TransformerClsFeaturesExtractor,
                "features_extractor_kwargs": {"encoder_name": args.encoder_name},
            }
        )
    ppo_model = PPO(
        policy_name,
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.ppo_epochs,
        gamma=args.gamma_rl,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=args.ppo_device,
    )

    run = None
    if args.wandb_mode != "disabled":
        if wandb is None:
            raise RuntimeError(
                "W&B mode is enabled but wandb is not installed. Install requirements first."
            )
        run_config = vars(args).copy()
        run_config["model_pool"] = model_pool
        run_config["reward_config"] = asdict(reward_config)
        run_config["train_records"] = len(train_records)
        run_config["val_records"] = len(val_records)
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=run_config,
        )

    eval_runner = None
    if len(val_records) > 0:
        eval_provider = OnlineOllamaProvider(provider_config)
        eval_runner = RouterEvaluator(
            records=val_records,
            observations=val_observations,
            model_pool=model_pool,
            provider=eval_provider,
            reward_config=reward_config,
        )

    callback = None
    if run is not None:
        callback = WandbMetricsCallback(
            run=run,
            model_pool=model_pool,
            eval_runner=eval_runner,
            output_dir=output_dir,
            log_every_steps=args.log_every_steps,
            eval_every_steps=args.eval_every_steps,
            checkpoint_metric=args.checkpoint_metric,
        )

    ppo_model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    final_model_path = output_dir / "router_policy_final"
    ppo_model.save(str(final_model_path))
    summary: Dict[str, Any] = {
        "final_model_path": str(final_model_path) + ".zip",
        "total_timesteps": args.total_timesteps,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "model_pool": model_pool,
        "encoder_training_mode": args.encoder_training_mode,
        "encoder_name": args.encoder_name,
        "encoder_max_length": args.encoder_max_length,
    }

    if eval_runner is not None:
        final_eval = eval_runner.evaluate(ppo_model)
        summary["final_eval"] = final_eval
        if run is not None and final_eval:
            run.log(final_eval)

    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if run is not None:
        artifact = wandb.Artifact("router-policy-final", type="model")
        artifact.add_file(str(final_model_path) + ".zip")
        artifact.add_file(str(config_path))
        artifact.add_file(str(summary_path))
        run.log_artifact(artifact)
        run.finish()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
