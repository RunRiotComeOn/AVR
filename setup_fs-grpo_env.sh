#!/bin/bash
# Set up the environment for FS-GRPO training with verl.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=${1:-verl}
TMPDIR=${TMPDIR:-/tmp/verl_setup}

mkdir -p "$TMPDIR"

echo "Creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.10 -y
ENV_PATH=$(conda run -n "$ENV_NAME" python -c "import sys; print(sys.prefix)")

echo "Installing vLLM"
TMPDIR="$TMPDIR" "$ENV_PATH/bin/pip" install --cache-dir "$TMPDIR/pip-cache" vllm==0.11.0

echo "Installing flash-attn"
TMPDIR="$TMPDIR" "$ENV_PATH/bin/pip" install --cache-dir "$TMPDIR/pip-cache" \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

echo "Installing training dependencies"
TMPDIR="$TMPDIR" "$ENV_PATH/bin/pip" install --cache-dir "$TMPDIR/pip-cache" \
    wandb hydra-core qwen-vl-utils accelerate datasets peft tensordict \
    pylatexenc liger-kernel dill

echo "Pinning transformers to the 4.x series"
TMPDIR="$TMPDIR" "$ENV_PATH/bin/pip" install --cache-dir "$TMPDIR/pip-cache" \
    "transformers>=4.51.0,<5.0.0"

echo "Installing huggingface_hub"
TMPDIR="$TMPDIR" "$ENV_PATH/bin/pip" install --cache-dir "$TMPDIR/pip-cache" \
    -U "huggingface_hub[cli]"

echo "Installing verl in editable mode"
TMPDIR="$TMPDIR" "$ENV_PATH/bin/pip" install --cache-dir "$TMPDIR/pip-cache" \
    -e "$REPO_DIR/verl"

if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "Logging into Weights & Biases"
    conda run -n "$ENV_NAME" wandb login "$WANDB_API_KEY"
fi

if [ -n "${HF_TOKEN:-}" ]; then
    echo "Logging into Hugging Face"
    conda run -n "$ENV_NAME" huggingface-cli login --token "$HF_TOKEN"
fi

echo "Environment setup completed."
echo "Activate it with: conda activate $ENV_NAME"
