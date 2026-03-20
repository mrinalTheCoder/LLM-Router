#!/usr/bin/env python3
"""Run llama3.2:1b on HuggingFace MMLU test split via Ollama."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

NANOSECONDS_TO_SECONDS = 1e-9


CHOICE_LABELS = ("A", "B", "C", "D")
ANSWER_RE = re.compile(r"\b([ABCD])\b")


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


def _truth_letter(example: Dict[str, Any]) -> str:
    answer = example["answer"]
    if isinstance(answer, int):
        return CHOICE_LABELS[answer]
    answer_text = str(answer).strip().upper()
    if answer_text in CHOICE_LABELS:
        return answer_text
    if answer_text.isdigit() and int(answer_text) in range(4):
        return CHOICE_LABELS[int(answer_text)]
    raise ValueError(f"Unsupported answer format: {answer!r}")


def _load_mmlu_rows_with_polars(
    dataset_name: str, dataset_config: str, split: str, limit: Optional[int]
) -> List[Dict[str, Any]]:
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: 'polars'. Install with `pip install polars` "
            "inside your llm-router environment."
        ) from exc

    parquet_path = f"hf://datasets/{dataset_name}/{dataset_config}/{split}-*.parquet"
    df = pl.read_parquet(parquet_path)
    if limit is not None:
        df = df.head(limit)
    return df.to_dicts()


def _sample_nvidia_gpu_power_watts(gpu_device: int) -> Optional[float]:
    command = [
        "nvidia-smi",
        "--query-gpu=power.draw",
        "--format=csv,noheader,nounits",
        "--id",
        str(gpu_device),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    power_text = lines[0]
    if power_text.upper() in {"N/A", "[NOT SUPPORTED]"}:
        return None
    try:
        return float(power_text)
    except ValueError:
        return None


class NvidiaPowerSampler:
    """Sample GPU power during generation and integrate energy."""

    def __init__(self, gpu_device: int = 0, interval_s: float = 0.1):
        self.gpu_device = gpu_device
        self.interval_s = max(interval_s, 0.02)
        self._samples: List[Tuple[float, float]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active = False

    def _take_sample(self) -> Optional[Tuple[float, float]]:
        power_watts = _sample_nvidia_gpu_power_watts(self.gpu_device)
        if power_watts is None:
            return None
        return (time.perf_counter(), power_watts)

    def _loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            sample = self._take_sample()
            if sample is not None:
                with self._lock:
                    self._samples.append(sample)
            next_tick += self.interval_s
            delay = next_tick - time.perf_counter()
            if delay > 0:
                self._stop_event.wait(delay)
            else:
                next_tick = time.perf_counter()

    def start(self) -> bool:
        if shutil.which("nvidia-smi") is None:
            return False
        first_sample = self._take_sample()
        if first_sample is None:
            return False
        with self._lock:
            self._samples.append(first_sample)
        self._active = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop_and_compute(self, elapsed_s: float) -> Tuple[Optional[float], Optional[float], int]:
        if not self._active:
            return (None, None, 0)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        final_sample = self._take_sample()
        if final_sample is not None:
            with self._lock:
                self._samples.append(final_sample)
        with self._lock:
            samples = list(self._samples)
        if not samples or elapsed_s <= 0:
            return (None, None, len(samples))

        samples.sort(key=lambda item: item[0])
        if len(samples) == 1:
            energy_joules = samples[0][1] * elapsed_s
        else:
            energy_joules = 0.0
            for (t1, p1), (t2, p2) in zip(samples, samples[1:]):
                dt = t2 - t1
                if dt <= 0:
                    continue
                energy_joules += 0.5 * (p1 + p2) * dt
        average_power = energy_joules / elapsed_s if elapsed_s > 0 else None
        return (energy_joules, average_power, len(samples))


def run_ollama_prompt(
    prompt: str,
    model: str,
    host: str = "http://localhost:11434",
    timeout_s: float = 300.0,
    gpu_layers: Optional[int] = 999,
    gpu_device: int = 0,
    power_sample_interval_s: float = 0.1,
) -> Dict[str, Any]:
    sampler = NvidiaPowerSampler(gpu_device=gpu_device, interval_s=power_sample_interval_s)
    sampler.start()
    start = time.perf_counter()

    options: Dict[str, Any] = {"main_gpu": gpu_device}
    if gpu_layers is not None:
        options["num_gpu"] = gpu_layers
    payload = {"model": model, "prompt": prompt, "stream": True, "options": options}
    request = urllib.request.Request(
        url=f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    response_parts: List[str] = []
    ttft_s: Optional[float] = None
    final_obj: Optional[Dict[str, Any]] = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunk = obj.get("response", "")
                if chunk:
                    if ttft_s is None:
                        ttft_s = time.perf_counter() - start
                    response_parts.append(chunk)
                if obj.get("done"):
                    final_obj = obj
                    break
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP error {err.code}: {body}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Could not connect to Ollama at {host}: {err}") from err

    end = time.perf_counter()
    total_latency_s = end - start
    gpu_energy_joules, gpu_average_power_watts, gpu_power_samples = sampler.stop_and_compute(
        total_latency_s
    )

    response_text = "".join(response_parts)
    eval_count = final_obj.get("eval_count") if final_obj else None
    prompt_eval_count = final_obj.get("prompt_eval_count") if final_obj else None
    eval_duration_s = (
        final_obj["eval_duration"] * NANOSECONDS_TO_SECONDS
        if final_obj and final_obj.get("eval_duration") is not None
        else None
    )
    prompt_eval_duration_s = (
        final_obj["prompt_eval_duration"] * NANOSECONDS_TO_SECONDS
        if final_obj and final_obj.get("prompt_eval_duration") is not None
        else None
    )
    total_duration_s = (
        final_obj["total_duration"] * NANOSECONDS_TO_SECONDS
        if final_obj and final_obj.get("total_duration") is not None
        else None
    )
    eval_tokens_per_s = (
        (eval_count / eval_duration_s)
        if eval_count is not None and eval_duration_s and eval_duration_s > 0
        else None
    )
    prompt_eval_tokens_per_s = (
        (prompt_eval_count / prompt_eval_duration_s)
        if prompt_eval_count is not None
        and prompt_eval_duration_s
        and prompt_eval_duration_s > 0
        else None
    )
    wall_tokens_per_s = (
        (eval_count / total_latency_s)
        if eval_count is not None and total_latency_s > 0
        else None
    )

    return {
        "response_text": response_text,
        "ttft_s": ttft_s,
        "total_latency_s": total_latency_s,
        "wall_tokens_per_s": wall_tokens_per_s,
        "eval_tokens_per_s": eval_tokens_per_s,
        "prompt_eval_tokens_per_s": prompt_eval_tokens_per_s,
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
        "eval_duration_s": eval_duration_s,
        "prompt_eval_duration_s": prompt_eval_duration_s,
        "total_duration_s": total_duration_s,
        "gpu_energy_joules": gpu_energy_joules,
        "gpu_average_power_watts": gpu_average_power_watts,
        "gpu_power_samples": gpu_power_samples,
        "energy_joules": gpu_energy_joules,
        "average_power_watts": gpu_average_power_watts,
        "energy_source": (
            f"nvidia_smi_gpu_{gpu_device}" if gpu_power_samples > 0 else "unavailable"
        ),
    }


def evaluate_mmlu(
    model: str = "llama3.2:1b",
    dataset_name: str = "cais/mmlu",
    dataset_config: str = "all",
    split: str = "test",
    limit: Optional[int] = None,
    host: str = "http://localhost:11434",
    timeout_s: float = 300.0,
    gpu_layers: Optional[int] = 999,
    gpu_device: int = 0,
    power_sample_interval_s: float = 0.1,
) -> Dict[str, Any]:
    rows = _load_mmlu_rows_with_polars(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        limit=limit,
    )
    run_size = len(rows)

    correct = 0
    parsed = 0
    sample_results: List[Dict[str, Any]] = []
    totals: Dict[str, float] = {
        "latency_s": 0.0,
        "ttft_s": 0.0,
        "wall_tokens_per_s": 0.0,
        "eval_tokens_per_s": 0.0,
        "energy_joules": 0.0,
        "average_power_watts": 0.0,
    }
    counts: Dict[str, int] = {key: 0 for key in totals}

    for idx, example in enumerate(rows):
        question = str(example["question"])
        choices = example["choices"]
        if not isinstance(choices, (list, tuple)) or len(choices) < 4:
            raise ValueError(f"Unexpected MMLU choices format at item {idx}: {choices!r}")

        prompt = build_prompt(question, choices[:4])
        metrics = run_ollama_prompt(
            prompt=prompt,
            model=model,
            host=host,
            timeout_s=timeout_s,
            gpu_layers=gpu_layers,
            gpu_device=gpu_device,
            power_sample_interval_s=power_sample_interval_s,
        )
        predicted = extract_answer_letter(metrics["response_text"])
        truth = _truth_letter(example)

        if predicted is not None:
            parsed += 1
            if predicted == truth:
                correct += 1

        for key, metric_key in (
            ("latency_s", "total_latency_s"),
            ("ttft_s", "ttft_s"),
            ("wall_tokens_per_s", "wall_tokens_per_s"),
            ("eval_tokens_per_s", "eval_tokens_per_s"),
            ("energy_joules", "energy_joules"),
            ("average_power_watts", "average_power_watts"),
        ):
            value = metrics.get(metric_key)
            if value is not None:
                totals[key] += float(value)
                counts[key] += 1

        sample_results.append(
            {
                "index": idx,
                "subject": example.get("subject"),
                "truth": truth,
                "predicted": predicted,
                "correct": predicted == truth if predicted is not None else False,
                "response_text": metrics["response_text"],
                "total_latency_s": metrics["total_latency_s"],
                "ttft_s": metrics["ttft_s"],
                "wall_tokens_per_s": metrics["wall_tokens_per_s"],
                "energy_joules": metrics["energy_joules"],
                "average_power_watts": metrics["average_power_watts"],
                "gpu_energy_joules": metrics["gpu_energy_joules"],
                "gpu_average_power_watts": metrics["gpu_average_power_watts"],
                "energy_source": metrics["energy_source"],
            }
        )

    averages = {
        key: (totals[key] / counts[key] if counts[key] > 0 else None) for key in totals
    }

    return {
        "model": model,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "evaluated_examples": run_size,
        "parsed_predictions": parsed,
        "correct_predictions": correct,
        "accuracy": (correct / run_size) if run_size else 0.0,
        "parse_rate": (parsed / run_size) if run_size else 0.0,
        "avg_latency_s": averages["latency_s"],
        "avg_ttft_s": averages["ttft_s"],
        "avg_wall_tokens_per_s": averages["wall_tokens_per_s"],
        "avg_eval_tokens_per_s": averages["eval_tokens_per_s"],
        "avg_energy_joules": averages["energy_joules"],
        "avg_power_watts": averages["average_power_watts"],
        "total_energy_joules": totals["energy_joules"] if counts["energy_joules"] > 0 else None,
        "samples": sample_results,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an Ollama model on HuggingFace MMLU test split."
    )
    parser.add_argument("--model", default="llama3.2:1b", help="Ollama model name")
    parser.add_argument("--dataset", default="cais/mmlu", help="HuggingFace dataset ID")
    parser.add_argument("--config", default="all", help="Dataset configuration")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument(
        "--host", default="http://localhost:11434", help="Ollama host URL"
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="Per-sample timeout in seconds"
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=999,
        help="Ollama num_gpu option (layers offloaded to GPU).",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="GPU index for Ollama option/main metrics sampling.",
    )
    parser.add_argument(
        "--power-sample-interval",
        type=float,
        default=0.1,
        help="Seconds between GPU power samples.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON file path. Prints to stdout when omitted.",
    )
    args = parser.parse_args()

    result = evaluate_mmlu(
        model=args.model,
        dataset_name=args.dataset,
        dataset_config=args.config,
        split=args.split,
        limit=args.limit,
        host=args.host,
        timeout_s=args.timeout,
        gpu_layers=args.gpu_layers,
        gpu_device=args.gpu_device,
        power_sample_interval_s=args.power_sample_interval,
    )

    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
    else:
        print(text)


if __name__ == "__main__":
    _cli()
