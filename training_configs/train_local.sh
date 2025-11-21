#!/bin/bash
# HDT Pre-training Script (Local GPU)
#
# This script replicates the HDT paper configuration.
# All hyperparameters are loaded from training_configs/hyperparams.json
#
# Paper Configuration (HDT-E):
# - Model: 12 encoder layers, 768 hidden, 12 heads = 109M params
# - Training: 24 hours on 1 GPU (academic budget, CRAMMING style)
# - Learning Rate: 1e-3 with AdamW
# - Batch Size: 2 with 16 gradient accumulation steps (effective: 32)
# - Context: 8192 tokens
# - Tokenizer: google-bert/bert-base-uncased (or google-t5/t5-base recommended)
# - Datasets: unarXive + Wikipedia + HUPD (streaming)

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR" || exit

# Create logs directory
mkdir -p logs

# Load hyperparameters
HYPERPARAMS="$SCRIPT_DIR/hyperparams.json"

# Source the hyperparameter loading function
source "$SCRIPT_DIR/load_hyperparams.sh"

# Load hyperparameters
load_hyperparams "$HYPERPARAMS"

# Add timestamp to save directory if not already present
if [[ ! "$SAVE_DIR" =~ [0-9]{8}_[0-9]{6} ]]; then
    SAVE_DIR="${SAVE_DIR}_$(date +%Y%m%d_%H%M%S)"
fi

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating it with uv sync..."
    uv sync
fi

# Activate environment
source .venv/bin/activate

# Load WandB API key if .env file exists
if [ -f .env ]; then
    echo "Loading WandB credentials from .env"
    export $(grep -v '^#' .env | xargs)
fi

# Disable wandb by default (will be re-enabled after loading hyperparams)
export WANDB_MODE="disabled"
export WANDB_SILENT="true"

# Auto-detect number of available GPUs for local training (before printing params)
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU may not be available."
    NUM_GPUS=1
else
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    if [ -n "$AVAILABLE_GPUS" ] && [ "$AVAILABLE_GPUS" -gt 0 ]; then
        NUM_GPUS=$AVAILABLE_GPUS
    else
        NUM_GPUS=1
    fi
fi

# Configure wandb mode based on hyperparameter
if [ "$WANDB_ENABLED" = "true" ]; then
    unset WANDB_MODE  # Allow normal online syncing
    unset WANDB_SILENT
    echo "WandB logging: enabled (online mode)"
else
    export WANDB_MODE="disabled"
    echo "WandB logging: disabled"
fi

# Print training information
echo "=================================================="
echo "Starting HDT pre-training on local GPU..."
echo "=================================================="
echo "Save directory: $SAVE_DIR"
echo "Tokenizer: $TOK_NAME"
echo "Encoder layers: $NUM_ENCODER_LAYERS"
echo "Decoder layers: $NUM_DECODER_LAYERS"
echo "Max input length: $MAX_INPUT_LENGTH"
echo "Max output length: $MAX_OUTPUT_LENGTH"
echo "MLM probability: $MLM_PROBABILITY"
echo "Learning rate: $LEARNING_RATE"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $ACCUMULATE_GRAD_BATCHES"
echo "Budget: $BUDGET hours"
echo "Number of GPUs: $NUM_GPUS (auto-detected)"
echo "Cache directory: $CACHE_DIR"
echo "Encoder only: $ENCODER_ONLY"
echo "WandB enabled: $WANDB_ENABLED"
echo "=================================================="

# Display GPU information
echo "Available GPUs:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --list-gpus
else
    echo "nvidia-smi not available"
fi
echo "=================================================="

# Check for existing checkpoints to resume from
LATEST_CKPT=$(
    find "$SAVE_DIR" -maxdepth 2 -type f -name 'last.ckpt' 2>/dev/null | head -n 1
)
if [ -n "$LATEST_CKPT" ]; then
    echo "Found existing checkpoint: $LATEST_CKPT"
    echo "Training will resume from this checkpoint"
    RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
else
    echo "No existing checkpoint found. Starting fresh training."
    RESUME_ARG=""
fi
echo "=================================================="

# Build command with all parameters
CMD="python pretrain.py \
    --save_dir $SAVE_DIR \
    --tok_name $TOK_NAME \
    --num_encoder_layers $NUM_ENCODER_LAYERS \
    --num_decoder_layers $NUM_DECODER_LAYERS \
    --max_input_length $MAX_INPUT_LENGTH \
    --max_output_length $MAX_OUTPUT_LENGTH \
    --mlm_probability $MLM_PROBABILITY \
    --lr $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --budget $BUDGET \
    --num_gpus $NUM_GPUS \
    --cache_dir $CACHE_DIR"

# Add encoder_only flag if set to true
if [ "$ENCODER_ONLY" = "true" ]; then
    CMD="$CMD --encoder_only"
fi

# Add wandb_enabled flag if set to true
if [ "$WANDB_ENABLED" = "true" ]; then
    CMD="$CMD --wandb_enabled"
fi

# Add gradient_checkpointing flag if set to true
if [ "$GRADIENT_CHECKPOINTING" = "true" ]; then
    CMD="$CMD --gradient_checkpointing"
fi

# Add resume argument if checkpoint exists
if [ -n "$RESUME_ARG" ]; then
    CMD="$CMD $RESUME_ARG"
fi

# Print the full command
echo "Executing command:"
echo "$CMD"
echo "=================================================="

# Execute training
eval "$CMD"

# Check exit status
EXIT_CODE=$?
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Model saved to: $SAVE_DIR"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=================================================="

exit $EXIT_CODE

