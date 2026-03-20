#!/usr/bin/env python3
"""Benchmark llama3.2:1b on MMLU test and export rich per-sample CSV telemetry."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
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
OLLAMA_CMD_RE = re.compile(r"(^|/|\s)ollama(\s|$)")


def _iso_utc(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat()


def _parse_float_field(value: str) -> Optional[float]:
    text = value.strip()
    if not text or text.upper() in {"N/A", "[NOT SUPPORTED]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_ps_time_to_seconds(text: str) -> Optional[float]:
    # ps TIME format examples: MM:SS, HH:MM:SS, DD-HH:MM:SS
    value = text.strip()
    if not value:
        return None
    day_part = 0
    if "-" in value:
        days, rest = value.split("-", 1)
        if not days.isdigit():
            return None
        day_part = int(days)
        value = rest
    parts = value.split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        return None
    try:
        return day_part * 86400 + int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


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


def build_prompt(question: str, choices: Sequence[str]) -> str:
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


def _snapshot_ollama_process_tree() -> Dict[str, Optional[float]]:
    command = ["ps", "-eo", "pid=,ppid=,rss=,time=,args="]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"cpu_time_s": None, "rss_bytes": None, "pid_count": None}

    procs: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        parts = text.split(None, 4)
        if len(parts) < 5:
            continue
        pid_text, ppid_text, rss_text, cpu_time_text, args = parts
        try:
            pid = int(pid_text)
            ppid = int(ppid_text)
            rss_kib = int(rss_text)
        except ValueError:
            continue
        procs.append(
            {
                "pid": pid,
                "ppid": ppid,
                "rss_bytes": rss_kib * 1024,
                "cpu_time_s": _parse_ps_time_to_seconds(cpu_time_text),
                "args": args,
            }
        )

    children: Dict[int, List[int]] = {}
    for proc in procs:
        children.setdefault(proc["ppid"], []).append(proc["pid"])
    roots = [proc["pid"] for proc in procs if OLLAMA_CMD_RE.search(proc["args"])]
    if not roots:
        return {"cpu_time_s": None, "rss_bytes": None, "pid_count": 0.0}

    proc_by_pid = {proc["pid"]: proc for proc in procs}
    stack = list(roots)
    in_tree: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in in_tree:
            continue
        in_tree.add(pid)
        for child in children.get(pid, []):
            stack.append(child)

    cpu_total = 0.0
    cpu_any = False
    rss_total = 0.0
    for pid in in_tree:
        proc = proc_by_pid.get(pid)
        if proc is None:
            continue
        rss_total += float(proc["rss_bytes"])
        cpu_time_s = proc.get("cpu_time_s")
        if cpu_time_s is not None:
            cpu_total += float(cpu_time_s)
            cpu_any = True

    return {
        "cpu_time_s": cpu_total if cpu_any else None,
        "rss_bytes": rss_total,
        "pid_count": float(len(in_tree)),
    }


def _compute_process_metrics(
    start_snapshot: Dict[str, Optional[float]],
    end_snapshot: Dict[str, Optional[float]],
    elapsed_s: float,
) -> Dict[str, Optional[float]]:
    start_cpu_time_s = start_snapshot.get("cpu_time_s")
    end_cpu_time_s = end_snapshot.get("cpu_time_s")
    cpu_time_delta_s = None
    if start_cpu_time_s is not None and end_cpu_time_s is not None:
        cpu_time_delta_s = max(0.0, end_cpu_time_s - start_cpu_time_s)
    cpu_percent = (
        (cpu_time_delta_s / elapsed_s) * 100.0
        if cpu_time_delta_s is not None and elapsed_s > 0
        else None
    )

    start_rss_bytes = start_snapshot.get("rss_bytes")
    end_rss_bytes = end_snapshot.get("rss_bytes")
    rss_delta_bytes = None
    if start_rss_bytes is not None and end_rss_bytes is not None:
        rss_delta_bytes = end_rss_bytes - start_rss_bytes
    rss_candidates = [v for v in (start_rss_bytes, end_rss_bytes) if v is not None]
    rss_peak_bytes = max(rss_candidates) if rss_candidates else None

    return {
        "cpu_time_start_s": start_cpu_time_s,
        "cpu_time_end_s": end_cpu_time_s,
        "cpu_time_delta_s": cpu_time_delta_s,
        "cpu_percent": cpu_percent,
        "cpu_effort_cpu_seconds": cpu_time_delta_s,
        "rss_start_bytes": start_rss_bytes,
        "rss_end_bytes": end_rss_bytes,
        "rss_delta_bytes": rss_delta_bytes,
        "rss_peak_bytes": rss_peak_bytes,
        "ollama_tree_pid_count_start": start_snapshot.get("pid_count"),
        "ollama_tree_pid_count_end": end_snapshot.get("pid_count"),
    }


def _query_nvidia_metrics(
    gpu_device: int, include_total_energy: bool
) -> Optional[Dict[str, Optional[float]]]:
    if include_total_energy:
        query = "power.draw,utilization.gpu,memory.used,total_energy_consumption"
    else:
        query = "power.draw,utilization.gpu,memory.used"

    command = [
        "nvidia-smi",
        "--query-gpu",
        query,
        "--format=csv,noheader,nounits",
        "--id",
        str(gpu_device),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None

    fields = [field.strip() for field in lines[0].split(",")]
    if include_total_energy and len(fields) >= 4:
        total_energy_mj = _parse_float_field(fields[3])
        return {
            "power_watts": _parse_float_field(fields[0]),
            "utilization_pct": _parse_float_field(fields[1]),
            "memory_used_mib": _parse_float_field(fields[2]),
            "total_energy_joules": (
                (total_energy_mj / 1000.0) if total_energy_mj is not None else None
            ),
        }
    if not include_total_energy and len(fields) >= 3:
        return {
            "power_watts": _parse_float_field(fields[0]),
            "utilization_pct": _parse_float_field(fields[1]),
            "memory_used_mib": _parse_float_field(fields[2]),
            "total_energy_joules": None,
        }
    return None


class NvidiaTelemetrySampler:
    """Sample GPU telemetry and compute power/energy/utilization metrics."""

    def __init__(self, gpu_device: int = 0, interval_s: float = 0.1):
        self.gpu_device = gpu_device
        self.interval_s = max(interval_s, 0.02)
        self._samples: List[Dict[str, Optional[float]]] = []
        self._use_total_energy_query = True
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active = False

    def _take_sample(self) -> Optional[Dict[str, Optional[float]]]:
        metrics = _query_nvidia_metrics(self.gpu_device, self._use_total_energy_query)
        if metrics is None and self._use_total_energy_query:
            self._use_total_energy_query = False
            metrics = _query_nvidia_metrics(self.gpu_device, self._use_total_energy_query)
        if metrics is None:
            return None
        return {
            "timestamp": time.perf_counter(),
            "power_watts": metrics.get("power_watts"),
            "utilization_pct": metrics.get("utilization_pct"),
            "memory_used_mib": metrics.get("memory_used_mib"),
            "total_energy_joules": metrics.get("total_energy_joules"),
        }

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

    def stop_and_compute(self, elapsed_s: float) -> Dict[str, Any]:
        if not self._active:
            return {
                "gpu_power_sample_count": 0.0,
                "gpu_power_samples_json": "[]",
                "gpu_power_avg_watts": None,
                "gpu_power_min_watts": None,
                "gpu_power_max_watts": None,
                "gpu_utilization_avg_pct": None,
                "gpu_vram_used_avg_mib": None,
                "gpu_total_energy_start_j": None,
                "gpu_total_energy_end_j": None,
                "gpu_total_energy_delta_j": None,
                "gpu_sampled_energy_joules": None,
                "gpu_energy_joules": None,
                "gpu_average_power_watts": None,
                "gpu_energy_source": "unavailable",
            }

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
            return {
                "gpu_power_sample_count": 0.0,
                "gpu_power_samples_json": "[]",
                "gpu_power_avg_watts": None,
                "gpu_power_min_watts": None,
                "gpu_power_max_watts": None,
                "gpu_utilization_avg_pct": None,
                "gpu_vram_used_avg_mib": None,
                "gpu_total_energy_start_j": None,
                "gpu_total_energy_end_j": None,
                "gpu_total_energy_delta_j": None,
                "gpu_sampled_energy_joules": None,
                "gpu_energy_joules": None,
                "gpu_average_power_watts": None,
                "gpu_energy_source": "unavailable",
            }

        samples.sort(key=lambda s: float(s["timestamp"] or 0.0))
        power_samples = [
            (float(s["timestamp"]), float(s["power_watts"]))
            for s in samples
            if s.get("timestamp") is not None and s.get("power_watts") is not None
        ]
        util_values = [
            float(s["utilization_pct"]) for s in samples if s.get("utilization_pct") is not None
        ]
        mem_values = [
            float(s["memory_used_mib"]) for s in samples if s.get("memory_used_mib") is not None
        ]
        total_energy_values = [
            float(s["total_energy_joules"])
            for s in samples
            if s.get("total_energy_joules") is not None
        ]

        sampled_energy_joules = None
        if power_samples:
            if len(power_samples) == 1:
                sampled_energy_joules = power_samples[0][1] * elapsed_s
            else:
                integral = 0.0
                for (t1, p1), (t2, p2) in zip(power_samples, power_samples[1:]):
                    dt_s = t2 - t1
                    if dt_s <= 0:
                        continue
                    integral += 0.5 * (p1 + p2) * dt_s
                sampled_energy_joules = integral

        total_energy_start_j = total_energy_values[0] if total_energy_values else None
        total_energy_end_j = total_energy_values[-1] if total_energy_values else None
        total_energy_delta_j = None
        if total_energy_start_j is not None and total_energy_end_j is not None:
            delta = total_energy_end_j - total_energy_start_j
            total_energy_delta_j = delta if delta >= 0 else None

        if total_energy_delta_j is not None:
            gpu_energy_joules = total_energy_delta_j
            energy_source = "nvidia_total_energy_delta"
        elif sampled_energy_joules is not None:
            gpu_energy_joules = sampled_energy_joules
            energy_source = "nvidia_sampled_power_integral"
        else:
            gpu_energy_joules = None
            energy_source = "unavailable"

        gpu_average_power_watts = (
            gpu_energy_joules / elapsed_s if gpu_energy_joules is not None and elapsed_s > 0 else None
        )
        power_values = [p for _, p in power_samples]
        power_json = json.dumps([round(p, 4) for p in power_values])

        return {
            "gpu_power_sample_count": float(len(power_samples)),
            "gpu_power_samples_json": power_json,
            "gpu_power_avg_watts": (
                (sum(power_values) / len(power_values)) if power_values else None
            ),
            "gpu_power_min_watts": min(power_values) if power_values else None,
            "gpu_power_max_watts": max(power_values) if power_values else None,
            "gpu_utilization_avg_pct": (
                (sum(util_values) / len(util_values)) if util_values else None
            ),
            "gpu_vram_used_avg_mib": (
                (sum(mem_values) / len(mem_values)) if mem_values else None
            ),
            "gpu_total_energy_start_j": total_energy_start_j,
            "gpu_total_energy_end_j": total_energy_end_j,
            "gpu_total_energy_delta_j": total_energy_delta_j,
            "gpu_sampled_energy_joules": sampled_energy_joules,
            "gpu_energy_joules": gpu_energy_joules,
            "gpu_average_power_watts": gpu_average_power_watts,
            "gpu_energy_source": energy_source,
        }


def run_ollama_prompt(
    prompt: str,
    model: str,
    host: str,
    timeout_s: float,
    gpu_layers: Optional[int],
    gpu_device: int,
    power_sample_interval_s: float,
) -> Dict[str, Any]:
    sampler = NvidiaTelemetrySampler(gpu_device=gpu_device, interval_s=power_sample_interval_s)
    sampler.start()

    start_wall_epoch = time.time()
    start_perf = time.perf_counter()
    start_proc = _snapshot_ollama_process_tree()

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
                        ttft_s = time.perf_counter() - start_perf
                    response_parts.append(chunk)
                if obj.get("done"):
                    final_obj = obj
                    break
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP error {err.code}: {body}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Could not connect to Ollama at {host}: {err}") from err

    end_perf = time.perf_counter()
    end_wall_epoch = time.time()
    total_latency_s = end_perf - start_perf
    end_proc = _snapshot_ollama_process_tree()

    process_metrics = _compute_process_metrics(start_proc, end_proc, total_latency_s)
    gpu_metrics = sampler.stop_and_compute(total_latency_s)

    response_text = "".join(response_parts)
    eval_count = final_obj.get("eval_count") if final_obj else None
    prompt_eval_count = final_obj.get("prompt_eval_count") if final_obj else None
    eval_duration_ns = final_obj.get("eval_duration") if final_obj else None
    prompt_eval_duration_ns = final_obj.get("prompt_eval_duration") if final_obj else None
    total_duration_ns = final_obj.get("total_duration") if final_obj else None

    eval_duration_s = (
        eval_duration_ns * NANOSECONDS_TO_SECONDS if eval_duration_ns is not None else None
    )
    prompt_eval_duration_s = (
        prompt_eval_duration_ns * NANOSECONDS_TO_SECONDS
        if prompt_eval_duration_ns is not None
        else None
    )
    total_duration_s = (
        total_duration_ns * NANOSECONDS_TO_SECONDS if total_duration_ns is not None else None
    )

    eval_tokens_per_s = (
        (eval_count / eval_duration_ns) * 1e9
        if eval_count is not None and eval_duration_ns and eval_duration_ns > 0
        else None
    )
    prompt_eval_tokens_per_s = (
        (prompt_eval_count / prompt_eval_duration_ns) * 1e9
        if prompt_eval_count is not None
        and prompt_eval_duration_ns
        and prompt_eval_duration_ns > 0
        else None
    )
    wall_tokens_per_s = (
        (eval_count / total_latency_s)
        if eval_count is not None and total_latency_s > 0
        else None
    )

    energy_joules = gpu_metrics.get("gpu_energy_joules")
    energy_per_output_token_j = (
        (energy_joules / eval_count)
        if energy_joules is not None and eval_count is not None and eval_count > 0
        else None
    )

    return {
        "request_start_timestamp_utc": _iso_utc(start_wall_epoch),
        "request_end_timestamp_utc": _iso_utc(end_wall_epoch),
        "request_start_epoch_s": start_wall_epoch,
        "request_end_epoch_s": end_wall_epoch,
        "model": model,
        "prompt_size_chars": len(prompt),
        "prompt_size_bytes": len(prompt.encode("utf-8")),
        "output_size_chars": len(response_text),
        "output_size_bytes": len(response_text.encode("utf-8")),
        "response_text": response_text,
        "ttft_s": ttft_s,
        "total_latency_s": total_latency_s,
        "wall_tokens_per_s": wall_tokens_per_s,
        "eval_tokens_per_s": eval_tokens_per_s,
        "prompt_eval_tokens_per_s": prompt_eval_tokens_per_s,
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
        "eval_duration_ns": eval_duration_ns,
        "prompt_eval_duration_ns": prompt_eval_duration_ns,
        "total_duration_ns": total_duration_ns,
        "eval_duration_s": eval_duration_s,
        "prompt_eval_duration_s": prompt_eval_duration_s,
        "total_duration_s": total_duration_s,
        "cpu_time_start_s": process_metrics["cpu_time_start_s"],
        "cpu_time_end_s": process_metrics["cpu_time_end_s"],
        "cpu_time_delta_s": process_metrics["cpu_time_delta_s"],
        "cpu_percent": process_metrics["cpu_percent"],
        "cpu_effort_cpu_seconds": process_metrics["cpu_effort_cpu_seconds"],
        "rss_start_bytes": process_metrics["rss_start_bytes"],
        "rss_end_bytes": process_metrics["rss_end_bytes"],
        "rss_delta_bytes": process_metrics["rss_delta_bytes"],
        "rss_peak_bytes": process_metrics["rss_peak_bytes"],
        "ollama_tree_pid_count_start": process_metrics["ollama_tree_pid_count_start"],
        "ollama_tree_pid_count_end": process_metrics["ollama_tree_pid_count_end"],
        "gpu_utilization_avg_pct": gpu_metrics["gpu_utilization_avg_pct"],
        "gpu_vram_used_avg_mib": gpu_metrics["gpu_vram_used_avg_mib"],
        "gpu_power_sample_count": gpu_metrics["gpu_power_sample_count"],
        "gpu_power_samples_json": gpu_metrics["gpu_power_samples_json"],
        "gpu_power_avg_watts": gpu_metrics["gpu_power_avg_watts"],
        "gpu_power_min_watts": gpu_metrics["gpu_power_min_watts"],
        "gpu_power_max_watts": gpu_metrics["gpu_power_max_watts"],
        "gpu_total_energy_start_j": gpu_metrics["gpu_total_energy_start_j"],
        "gpu_total_energy_end_j": gpu_metrics["gpu_total_energy_end_j"],
        "gpu_total_energy_delta_j": gpu_metrics["gpu_total_energy_delta_j"],
        "gpu_sampled_energy_joules": gpu_metrics["gpu_sampled_energy_joules"],
        "gpu_energy_joules": gpu_metrics["gpu_energy_joules"],
        "gpu_average_power_watts": gpu_metrics["gpu_average_power_watts"],
        "energy_joules": energy_joules,
        "average_power_watts": gpu_metrics["gpu_average_power_watts"],
        "energy_per_output_token_j": energy_per_output_token_j,
        "energy_source": gpu_metrics["gpu_energy_source"],
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
        prompt = build_prompt(str(example.get("question", "")), choices[:4])
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
                "request_start_timestamp_utc": metrics["request_start_timestamp_utc"],
                "request_end_timestamp_utc": metrics["request_end_timestamp_utc"],
                "request_start_epoch_s": metrics["request_start_epoch_s"],
                "request_end_epoch_s": metrics["request_end_epoch_s"],
                "model": metrics["model"],
                "prompt_size_chars": metrics["prompt_size_chars"],
                "prompt_size_bytes": metrics["prompt_size_bytes"],
                "output_size_chars": metrics["output_size_chars"],
                "output_size_bytes": metrics["output_size_bytes"],
                "prompt_eval_count": metrics["prompt_eval_count"],
                "eval_count": metrics["eval_count"],
                "prompt_eval_duration_ns": metrics["prompt_eval_duration_ns"],
                "eval_duration_ns": metrics["eval_duration_ns"],
                "total_duration_ns": metrics["total_duration_ns"],
                "prompt_eval_duration_s": metrics["prompt_eval_duration_s"],
                "eval_duration_s": metrics["eval_duration_s"],
                "total_duration_s": metrics["total_duration_s"],
                "ttft_s": metrics["ttft_s"],
                "total_latency_s": metrics["total_latency_s"],
                "tokens_per_sec": metrics["eval_tokens_per_s"],
                "wall_tokens_per_s": metrics["wall_tokens_per_s"],
                "prompt_eval_tokens_per_s": metrics["prompt_eval_tokens_per_s"],
                "cpu_time_start_s": metrics["cpu_time_start_s"],
                "cpu_time_end_s": metrics["cpu_time_end_s"],
                "cpu_time_delta_s": metrics["cpu_time_delta_s"],
                "cpu_percent": metrics["cpu_percent"],
                "cpu_effort_cpu_seconds": metrics["cpu_effort_cpu_seconds"],
                "rss_start_bytes": metrics["rss_start_bytes"],
                "rss_end_bytes": metrics["rss_end_bytes"],
                "rss_delta_bytes": metrics["rss_delta_bytes"],
                "rss_peak_bytes": metrics["rss_peak_bytes"],
                "ollama_tree_pid_count_start": metrics["ollama_tree_pid_count_start"],
                "ollama_tree_pid_count_end": metrics["ollama_tree_pid_count_end"],
                "gpu_utilization_avg_pct": metrics["gpu_utilization_avg_pct"],
                "gpu_vram_used_avg_mib": metrics["gpu_vram_used_avg_mib"],
                "gpu_power_sample_count": metrics["gpu_power_sample_count"],
                "gpu_power_samples_json": metrics["gpu_power_samples_json"],
                "gpu_power_avg_watts": metrics["gpu_power_avg_watts"],
                "gpu_power_min_watts": metrics["gpu_power_min_watts"],
                "gpu_power_max_watts": metrics["gpu_power_max_watts"],
                "gpu_total_energy_start_j": metrics["gpu_total_energy_start_j"],
                "gpu_total_energy_end_j": metrics["gpu_total_energy_end_j"],
                "gpu_total_energy_delta_j": metrics["gpu_total_energy_delta_j"],
                "gpu_sampled_energy_joules": metrics["gpu_sampled_energy_joules"],
                "gpu_energy_joules": metrics["gpu_energy_joules"],
                "gpu_average_power_watts": metrics["gpu_average_power_watts"],
                "energy_joules": metrics["energy_joules"],
                "energy_per_output_token_j": metrics["energy_per_output_token_j"],
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
            "Benchmark llama3.2:1b on MMLU and write CSV with correctness, latency, "
            "CPU, memory, and GPU energy telemetry."
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
        help="Seconds between nvidia-smi telemetry samples.",
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
