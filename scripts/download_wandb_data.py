"""Download all run data from the LLM-Router W&B project.

Saves:
  - wandb_data/runs_summary.csv   — one row per run (config + summary metrics)
  - wandb_data/runs_history/      — one CSV per run with step-level history
  - wandb_data/runs_meta.json     — full metadata for all runs

Usage:
  python scripts/download_wandb_data.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import wandb

ENTITY = "llm-router"
PROJECT = "LLM-Router"
OUT_DIR = Path("wandb_data")
HISTORY_DIR = OUT_DIR / "runs_history"


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    HISTORY_DIR.mkdir(exist_ok=True)

    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    print(f"Found {len(runs)} runs in {ENTITY}/{PROJECT}")

    summary_rows = []
    meta_list = []

    for i, run in enumerate(runs):
        print(f"  [{i+1}/{len(runs)}] {run.name} ({run.id}) state={run.state}")

        # Flatten config + summary into one row
        row = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": str(run.created_at),
            "sweep_id": run.sweep.id if run.sweep else None,
            "sweep_name": run.sweep.name if run.sweep else None,
        }
        row.update(flatten(dict(run.config), "config"))
        row.update(flatten(dict(run.summary), "summary"))
        summary_rows.append(row)

        # Full metadata
        meta_list.append({
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": str(run.created_at),
            "sweep_id": run.sweep.id if run.sweep else None,
            "config": dict(run.config),
            "summary": {k: v for k, v in run.summary.items() if not k.startswith("_")},
            "tags": list(run.tags),
            "notes": run.notes,
        })

        # Step-level history
        try:
            history = run.history(samples=10000, pandas=True)
            if not history.empty:
                history.to_csv(HISTORY_DIR / f"{run.id}.csv", index=False)
        except Exception as exc:
            print(f"    Warning: could not fetch history for {run.id}: {exc}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "runs_summary.csv", index=False)
    print(f"\nSaved {len(summary_rows)} rows -> {OUT_DIR / 'runs_summary.csv'}")

    with open(OUT_DIR / "runs_meta.json", "w") as f:
        json.dump(meta_list, f, indent=2, default=str)
    print(f"Saved metadata -> {OUT_DIR / 'runs_meta.json'}")

    history_files = list(HISTORY_DIR.glob("*.csv"))
    print(f"Saved {len(history_files)} history files -> {HISTORY_DIR}/")


if __name__ == "__main__":
    main()
