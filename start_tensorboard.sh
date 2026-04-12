#!/bin/bash
# Start TensorBoard for training monitoring.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-verl}"
LOGDIR="${1:-${LOGDIR:-$REPO_DIR/verl/tensorboard_log}}"
PORT="${PORT:-6006}"

if [ ! -d "$LOGDIR" ]; then
    echo "ERROR: Log directory not found at $LOGDIR"
    echo "Pass a log directory as the first argument or set LOGDIR."
    exit 1
fi

echo "Starting TensorBoard..."
echo "Log directory: $LOGDIR"
echo "Port: $PORT"
echo ""
echo "Access TensorBoard at:"
echo "  Local: http://localhost:$PORT"
echo "  Remote: Use SSH port forwarding if needed"
echo ""
echo "SSH port forwarding example:"
echo "  ssh -L $PORT:localhost:$PORT <username>@<server-address>"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo "=========================================="
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Start TensorBoard
tensorboard --logdir="$LOGDIR" --port="$PORT" --bind_all
