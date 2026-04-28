#!/bin/bash
set -euo pipefail

SWEEP_CONFIG="${SWEEP_CONFIG:-scripts/wandb_sweep_router_ppo_focused.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-llm-router}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
AGENT_COUNT="${AGENT_COUNT:-20}"
SWEEP_ID="${SWEEP_ID:-}"

if ! command -v wandb >/dev/null 2>&1; then
    echo "wandb CLI is required. Install dependencies first." >&2
    exit 1
fi

if [[ ! -f "$SWEEP_CONFIG" ]]; then
    echo "Sweep config not found: $SWEEP_CONFIG" >&2
    exit 1
fi

if [[ -z "$SWEEP_ID" ]]; then
    if [[ -n "$WANDB_ENTITY" ]]; then
        sweep_output="$(wandb sweep --project "$WANDB_PROJECT" --entity "$WANDB_ENTITY" "$SWEEP_CONFIG")"
    else
        sweep_output="$(wandb sweep --project "$WANDB_PROJECT" "$SWEEP_CONFIG")"
    fi
    printf '%s\n' "$sweep_output"

    SWEEP_ID="$(printf '%s\n' "$sweep_output" | awk '/wandb agent/{print $NF; exit}')"
    if [[ -z "$SWEEP_ID" ]]; then
        echo "Unable to parse sweep id from wandb output. Set SWEEP_ID manually." >&2
        exit 1
    fi
fi

echo "Starting wandb agent for sweep: $SWEEP_ID"
wandb agent --count "$AGENT_COUNT" "$SWEEP_ID"
