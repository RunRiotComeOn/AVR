#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-$REPO_DIR/LLaMA-Factory}"
ENV_NAME="${ENV_NAME:-llamafactory}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$REPO_DIR/train_configs/qwen3vl_2b_full_sft_all.yaml}"

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate environment
echo "Activating conda environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CLI: $(which llamafactory-cli)"

if [ ! -d "$LLAMA_FACTORY_DIR" ]; then
    echo "ERROR: LLaMA-Factory directory not found at $LLAMA_FACTORY_DIR"
    exit 1
fi

if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "ERROR: Training config not found at $TRAIN_CONFIG"
    exit 1
fi

# Navigate to LLaMA-Factory directory
cd "$LLAMA_FACTORY_DIR"

# Set GPUs unless already provided by the caller
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Run training
echo "Starting SFT training..."
FORCE_TORCHRUN=1 llamafactory-cli train "$TRAIN_CONFIG"

echo "Training completed!"
