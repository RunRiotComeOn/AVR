#!/usr/bin/env bash
# Merge an FSDP checkpoint to Hugging Face format.

set -e

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <checkpoint_dir>"
    exit 1
fi

CHECKPOINT_DIR="$1"

if [ ! -d "$CHECKPOINT_DIR/actor" ]; then
    echo "ERROR: actor checkpoint not found at $CHECKPOINT_DIR/actor"
    exit 1
fi

echo "==================================="
echo "Merge FSDP Checkpoint"
echo "==================================="
echo "Checkpoint directory: $CHECKPOINT_DIR"

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CHECKPOINT_DIR}/actor" \
    --target_dir "${CHECKPOINT_DIR}/actor_merged"

echo ""
echo "Merge completed! Model saved to: ${CHECKPOINT_DIR}/actor_merged"
echo ""
