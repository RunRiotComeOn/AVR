#!/bin/bash
set -e

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate environment
echo "Activating llamafactory environment..."
conda activate llamafactory

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CLI: $(which llamafactory-cli)"

# Navigate to LLaMA-Factory directory
cd /nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory

# Set GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Run training
echo "Starting all-GPU training..."
FORCE_TORCHRUN=1 llamafactory-cli train ../train_configs/qwen3vl_2b_full_sft_all.yaml

echo "Training completed!"
