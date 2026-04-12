#!/bin/bash
# Start TensorBoard for FS-GRPO training monitoring

# FS-GRPO training log directory (from train-fs-grpo.sh)
# LOGDIR="/nas03/yixuh/vlm-adaptive-resoning/verl/tensorboard_log"
LOGDIR="/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/saves/qwen2_5vl-3b/phase/phase2/runs"

# Port for TensorBoard (change if needed)
PORT=6006

echo "🔍 Starting TensorBoard..."
echo "📂 Log directory: $LOGDIR"
echo "🌐 Port: $PORT"
echo ""
echo "⚡ Access TensorBoard at:"
echo "   Local: http://localhost:$PORT"
echo "   Remote: Use SSH port forwarding (see below)"
echo ""
echo "🔗 SSH Port Forwarding Command (run on your local machine):"
echo "   ssh -L $PORT:localhost:$PORT <username>@<server-address>"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo "=========================================="
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate verl

# Start TensorBoard
tensorboard --logdir=$LOGDIR --port=$PORT --bind_all
