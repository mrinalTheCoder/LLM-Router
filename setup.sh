#!/bin/bash

# Exit on error
set -e

echo "Loading Anaconda module..."
# Attempt to load anaconda3 if module command exists
if command -v module &> /dev/null; then
    module load anaconda3 || echo "Warning: 'module load anaconda3' returned non-zero. Continuing..."
else
    echo "'module' command not found, skipping 'module load anaconda3'."
fi

# Initialize conda for this script
# This is often necessary because conda activate is a shell function
eval "$(conda shell.bash hook)"

ENV_NAME="llm-router"

# Check if the environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME'..."
    # Create with Python 3.10 (common stable version for LLM dependecies)
    conda create -n "$ENV_NAME" python=3.10 -y
fi

echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found!"
fi



echo "==========================================="
echo "Setup complete!"
echo ""
echo "To use the environment, run:"
echo "  conda activate $ENV_NAME"
echo "==========================================="
