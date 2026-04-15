#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --model <model_name>"
}

MODEL_NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            shift
            if [[ $# -eq 0 ]]; then
                echo "Error: --model requires a value." >&2
                usage
                exit 1
            fi
            MODEL_NAME="$1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'." >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_NAME" ]]; then
    echo "Error: --model is required." >&2
    usage
    exit 1
fi

OUTPUT_DIR="${HOME}/scratch/model_outputs"
mkdir -p "$OUTPUT_DIR"

for SPLIT in test validation; do
    OUTPUT_CSV="${OUTPUT_DIR}/${MODEL_NAME}-${SPLIT}.csv"
    echo "Running split '${SPLIT}' -> ${OUTPUT_CSV}"
    python "scripts/benchmark_mmlu_csv.py" \
        --model "$MODEL_NAME" \
        --split "$SPLIT" \
        --output-csv "$OUTPUT_CSV"
done
