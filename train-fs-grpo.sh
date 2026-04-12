#!/bin/bash
# FS-GRPO training with adaptive reward.
# Uses an intra-group diversity bonus that decays over training steps.
# Early training: encourage diverse format exploration across the N responses per prompt.
# Late training: diversity pressure fades, quality/correctness dominates.

set -e

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Default parameters
export CUDA_VISIBLE_DEVICES=4,5,6,7
SFT_MODEL=""
DATA_DIR="fs-grpo_data_new"
OUTPUT_DIR="saves/qwen3vl-4b/fs-grpo/adaptive_reward"
NUM_GPUS=4
ENGINE="vllm"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sft_model)
            SFT_MODEL="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --sft_model <path>     Path to SFT checkpoint"
            echo "  --data_dir <path>      FS-GRPO data directory"
            echo "  --output_dir <path>    Output directory"
            echo "  --gpus <num>           Number of GPUs (default: 8)"
            echo "  --engine <vllm|sglang> Inference engine (default: vllm)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Auto-detect SFT model
if [ -z "$SFT_MODEL" ]; then
    if [ -d "LLaMA-Factory/saves/qwen2_5vl-3b/full/vision_sr1_1_epoch_no_st" ]; then
        SFT_MODEL="LLaMA-Factory/saves/qwen2_5vl-3b/full/vision_sr1_1_epoch_no_st"
        echo -e "${YELLOW}Auto-detected SFT model: $SFT_MODEL${NC}"
    elif [ -d "LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_9k" ]; then
        SFT_MODEL="LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_9k"
        echo -e "${YELLOW}Auto-detected SFT model: $SFT_MODEL${NC}"
    else
        echo -e "${RED}ERROR: SFT model not specified and no default found${NC}"
        echo "Please specify --sft_model or ensure a checkpoint exists"
        exit 1
    fi
fi

# Validate paths
if [ ! -d "$SFT_MODEL" ]; then
    echo -e "${RED}ERROR: SFT model not found at $SFT_MODEL${NC}"
    exit 1
fi

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo -e "${RED}ERROR: FS-GRPO training data not found at $DATA_DIR/train.parquet${NC}"
    echo "Please prepare the FS-GRPO parquet data first"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FS-GRPO Training - Adaptive Reward${NC}"
echo -e "${GREEN}  (with Diversity Bonus)${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "SFT Model: ${YELLOW}$SFT_MODEL${NC}"
echo -e "Data Directory: ${YELLOW}$DATA_DIR${NC}"
echo -e "Output Directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Number of GPUs: ${YELLOW}$NUM_GPUS${NC}"
echo -e "Inference Engine: ${YELLOW}$ENGINE${NC}"
echo -e "Reward Manager: ${YELLOW}adaptive_reward${NC}"
echo -e "Diversity: ${YELLOW}weight=0.5, decay=cosine${NC}"
echo -e "${GREEN}========================================${NC}"

# Install verl if needed
if ! python -c "import verl" 2>/dev/null; then
    echo -e "${YELLOW}Installing verl...${NC}"
    cd verl
    pip install -e .
    cd ..
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/verl"

# NCCL settings
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22
export NCCL_BLOCKING_WAIT=1

# PyTorch distributed timeout
export TORCH_DISTRIBUTED_TIMEOUT=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# VLLM settings
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo -e "${GREEN}Starting FS-GRPO training with diversity bonus...${NC}"

cd verl

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=../$DATA_DIR/train.parquet \
    data.val_files=../$DATA_DIR/val.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.image_key=images \
    data.prompt_key=prompt \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=../$SFT_MODEL \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.reward_manager=adaptive_reward \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard","wandb"]' \
    trainer.project_name='verl_fs-grpo_adaptive_reward' \
    trainer.experiment_name='qwen3vl_2b_adaptive_reward' \
    trainer.default_local_dir=../$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.resume_mode=auto \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=2

cd ..

echo -e "${GREEN}Training completed!${NC}"
echo -e "Model saved to: ${YELLOW}$OUTPUT_DIR${NC}"
