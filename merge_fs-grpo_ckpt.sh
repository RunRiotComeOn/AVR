#!/usr/bin/env bash
# Merge FSDP checkpoint to HuggingFace format and evaluate on OCR-VQA

set -e

CHECKPOINT_DIR="/nas03/yixuh/vlm-adaptive-resoning/saves/qwen2_5vl-3b/fs-grpo/adaptive_reward_full/global_step_220"

echo "==================================="
echo "Merge FSDP Checkpoint"
echo "==================================="

# Merge the FSDP checkpoint to HuggingFace format
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CHECKPOINT_DIR}/actor" \
    --target_dir "${CHECKPOINT_DIR}/actor_merged"

echo ""
echo "Merge completed! Model saved to: ${CHECKPOINT_DIR}/actor_merged"
echo ""
