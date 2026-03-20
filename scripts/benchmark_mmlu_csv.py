#!/usr/bin/env python3
"""Benchmark llama3.2:1b on MMLU test and export per-sample CSV metrics."""

from __future__ import annotations

import argparse
import csv
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
SIMPLE_ANSWER_RE = re.compile(r"^\s*([ABCD])\s*$", re.IGNORECASE)


def _load_mmlu_rows_with_polars(
    dataset_name: str, dataset_config: str, split: str, limit: Optional[int]
) -> List[Dict[str, Any]]:
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: 'polars'. Install with `pip install polars`."
        ) from exc

    parquet_path = f"hf://datasets/{dataset_name}/{dataset_config}/{split}-*.parquet"
    frame = pl.read_parquet(parquet_path)
    if limit is not None:
        frame = frame.head(limit)
    return frame.to_dicts()


def _truth_letter(answer: Any) -> str:
    if isinstance(answer, int):
        if answer not in range(4):
            raise ValueError(f"Unexpected numeric answer index: {answer}")
        return CHOICE_LABELS[answer]
    answer_text = str(answer).strip().upper()
    if answer_text in CHOICE_LABELS:
        return answer_text
    if answer_text.isdigit() and int(answer_text) in range(4):
        return CHOICE_LABELS[int(answer_text)]
    raise ValueError(f"Unsupported answer value: {answer!r}")


def build_reasoning_prompt(question: str, choices: Sequence[str]) -> str:
    lines = [
        "You are solving a multiple-choice question.",
        "Respond with only one character: A, B, C, or D.",
        "Do not include any explanation or extra text.",
        "",
        f"Question: {question}",
    ]
    for label, choice in zip(CHOICE_LABELS, choices):
        lines.append(f"{label}. {choice}")
    lines.extend(["", "Final answer:"])
    return "\n".join(lines)


def _parse_simple_answer(text: str) -> Tuple[Optional[str], bool]:
    match = SIMPLE_ANSWER_RE.match(text)
    if not match:
        return (None, False)
    return (match.group(1).upper(), True)


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
    host: str,
    timeout_s: float,
    gpu_layers: Optional[int],
    gpu_device: int,
    power_sample_interval_s: float,
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
        "energy_source": f"nvidia_smi_gpu_{gpu_device}" if gpu_power_samples > 0 else "unavailable",
    }


def benchmark_to_csv(
    output_csv: str,
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
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: 'tqdm'. Install with `pip install tqdm`."
        ) from exc

    rows = _load_mmlu_rows_with_polars(dataset_name, dataset_config, split, limit)
    csv_rows: List[Dict[str, Any]] = []
    correct = 0
    valid_answer_format = 0

    progress = tqdm(rows, total=len(rows), desc="Benchmarking MMLU", unit="q")
    for idx, example in enumerate(progress):
        choices = example.get("choices")
        if not isinstance(choices, (list, tuple)) or len(choices) < 4:
            raise ValueError(f"Unexpected choices at row {idx}: {choices!r}")
        truth = _truth_letter(example.get("answer"))
        prompt = build_reasoning_prompt(str(example.get("question", "")), choices[:4])
        metrics = run_ollama_prompt(
            prompt=prompt,
            model=model,
            host=host,
            timeout_s=timeout_s,
            gpu_layers=gpu_layers,
            gpu_device=gpu_device,
            power_sample_interval_s=power_sample_interval_s,
        )
        predicted, is_valid_format = _parse_simple_answer(metrics["response_text"])
        is_correct = bool(predicted == truth) if is_valid_format else False
        if is_valid_format:
            valid_answer_format += 1
        if is_correct:
            correct += 1

        progress.set_postfix(
            correct=correct,
            wrong=(idx + 1 - correct),
            valid_format=valid_answer_format,
        )

        csv_rows.append(
            {
                "index": idx,
                "subject": example.get("subject"),
                "question": example.get("question"),
                "choice_a": choices[0],
                "choice_b": choices[1],
                "choice_c": choices[2],
                "choice_d": choices[3],
                "truth": truth,
                "predicted_final_answer": predicted,
                "answer_format_valid": is_valid_format,
                "answer_format_failure": not is_valid_format,
                "result": "correct" if is_correct else "wrong",
                "is_correct": is_correct,
                "ttft_s": metrics["ttft_s"],
                "total_latency_s": metrics["total_latency_s"],
                "wall_tokens_per_s": metrics["wall_tokens_per_s"],
                "eval_tokens_per_s": metrics["eval_tokens_per_s"],
                "prompt_eval_tokens_per_s": metrics["prompt_eval_tokens_per_s"],
                "eval_count": metrics["eval_count"],
                "prompt_eval_count": metrics["prompt_eval_count"],
                "eval_duration_s": metrics["eval_duration_s"],
                "prompt_eval_duration_s": metrics["prompt_eval_duration_s"],
                "total_duration_s": metrics["total_duration_s"],
                "energy_joules": metrics["energy_joules"],
                "average_power_watts": metrics["average_power_watts"],
                "gpu_energy_joules": metrics["gpu_energy_joules"],
                "gpu_average_power_watts": metrics["gpu_average_power_watts"],
                "gpu_power_samples": metrics["gpu_power_samples"],
                "energy_source": metrics["energy_source"],
                "model_output": metrics["response_text"],
            }
        )

    if not csv_rows:
        raise RuntimeError("No rows loaded from dataset.")

    fieldnames = list(csv_rows[0].keys())
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    total = len(csv_rows)
    return {
        "output_csv": output_csv,
        "evaluated_examples": total,
        "correct_predictions": correct,
        "accuracy": correct / total if total else 0.0,
        "valid_answer_format_predictions": valid_answer_format,
        "valid_answer_format_rate": valid_answer_format / total if total else 0.0,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark llama3.2:1b on MMLU and write CSV with correctness, latency, and power metrics."
        )
    )
    parser.add_argument("--output-csv", default="mmlu_benchmark.csv", help="Output CSV path.")
    parser.add_argument("--model", default="llama3.2:1b", help="Ollama model name.")
    parser.add_argument("--dataset", default="cais/mmlu", help="Dataset name.")
    parser.add_argument("--config", default="all", help="Dataset config.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples.")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds.")
    parser.add_argument("--gpu-layers", type=int, default=999, help="Ollama num_gpu option.")
    parser.add_argument("--gpu-device", type=int, default=0, help="GPU device index.")
    parser.add_argument(
        "--power-sample-interval",
        type=float,
        default=0.1,
        help="Seconds between nvidia-smi power samples.",
    )
    args = parser.parse_args()

    summary = benchmark_to_csv(
        output_csv=args.output_csv,
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
