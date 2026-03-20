#!/usr/bin/env python3
"""Run llama3.2:1b on HuggingFace MMLU test split via Ollama."""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ollama_benchmark import run_ollama_prompt


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
