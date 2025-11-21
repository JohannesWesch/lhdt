#!/bin/bash
# HDT Pre-training Script (SLURM Cluster)
#
# This script replicates the HDT paper configuration on a GPU cluster.
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
#
# IMPORTANT: For paper replication, set num_gpus=1 and budget=24 in hyperparams.json
#            This cluster config shows 4 GPUs for faster training if desired.

#SBATCH --job-name=hdt-pretrain
#SBATCH --partition=gpu_h100_il
#SBATCH --mem=510000mb
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/training_hdt_%j.out
#SBATCH --error=logs/training_hdt_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module load devel/cuda/12.8

# Ensure we run from the submission directory so relative paths resolve
cd "$SLURM_SUBMIT_DIR" || exit

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating it with uv sync..."
    uv sync
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Load WandB API key if .env file exists
if [ -f .env ]; then
    echo "Loading WandB credentials from .env"
    export $(grep -v '^#' .env | xargs)
fi

# Disable wandb by default (will be re-enabled after loading hyperparams)
export WANDB_MODE="disabled"
export WANDB_SILENT="true"

# Load hyperparameters
# Check if we are in the parent directory (running from repo root) or inside training_configs
if [ -f "$SLURM_SUBMIT_DIR/training_configs/load_hyperparams.sh" ]; then
    CONFIG_DIR="$SLURM_SUBMIT_DIR/training_configs"
elif [ -f "$SLURM_SUBMIT_DIR/load_hyperparams.sh" ]; then
    CONFIG_DIR="$SLURM_SUBMIT_DIR"
else
    echo "Error: Could not locate load_hyperparams.sh in $SLURM_SUBMIT_DIR or $SLURM_SUBMIT_DIR/training_configs"
    exit 1
fi

HYPERPARAMS="$CONFIG_DIR/hyperparams.json"

# Source the hyperparameter loading function
source "$CONFIG_DIR/load_hyperparams.sh"

# Load hyperparameters
load_hyperparams "$HYPERPARAMS"

# Add timestamp to save directory if not already present
if [[ ! "$SAVE_DIR" =~ [0-9]{8}_[0-9]{6} ]]; then
    SAVE_DIR="${SAVE_DIR}_$(date +%Y%m%d_%H%M%S)"
fi

# Configure wandb mode based on hyperparameter
if [ "$WANDB_ENABLED" = "true" ]; then
    export WANDB_MODE="offline"  # Use offline mode for cluster (sync later)
    unset WANDB_SILENT
    echo "WandB logging: enabled (offline mode - sync after training)"
else
    export WANDB_MODE="disabled"
    echo "WandB logging: disabled"
fi

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "Number of tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Time limit: $SLURM_TIME_LIMIT"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Start Time: $(date)"
echo "=================================================="

# Set environment variables for distributed training
MASTER_ADDR=$(hostname)
export MASTER_ADDR
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Print distributed training environment variables
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Print CUDA information
nvidia-smi

# Print training configuration
echo "=================================================="
echo "Training Configuration (from $HYPERPARAMS):"
echo "  Save Directory: $SAVE_DIR"
echo "  Tokenizer: $TOK_NAME"
echo "  Encoder Layers: $NUM_ENCODER_LAYERS"
echo "  Decoder Layers: $NUM_DECODER_LAYERS"
echo "  Max Input Length: $MAX_INPUT_LENGTH"
echo "  Max Output Length: $MAX_OUTPUT_LENGTH"
echo "  MLM Probability: $MLM_PROBABILITY"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $ACCUMULATE_GRAD_BATCHES"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Budget: $BUDGET hours"
echo "  Cache Directory: $CACHE_DIR"
echo "  Encoder Only: $ENCODER_ONLY"
echo "=================================================="

# Check for existing checkpoints to resume from
echo "=================================================="
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

# Build training command with all parameters
# Use srun for proper distributed training with SLURM
CMD="srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE python pretrain.py \
    --save_dir $SAVE_DIR \
    --tok_name $TOK_NAME \
    --num_encoder_layers $NUM_ENCODER_LAYERS \
    --num_decoder_layers $NUM_DECODER_LAYERS \
    --max_input_length $MAX_INPUT_LENGTH \
    --max_output_length $MAX_OUTPUT_LENGTH \
    --mlm_probability $MLM_PROBABILITY \
    --batch_size $BATCH_SIZE \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --lr $LEARNING_RATE \
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

# Run training with srun for proper distributed execution
echo "Starting HDT pre-training..."
echo "Command: $CMD"
eval "$CMD"

# Check exit status
if [ $? -eq 0 ]; then
    echo "=================================================="
    echo "Training completed successfully!"
    echo "Model saved to: $SAVE_DIR"
    echo "End Time: $(date)"
    echo "=================================================="
else
    echo "=================================================="
    echo "Training failed with error code: $?"
    echo "End Time: $(date)"
    echo "=================================================="
    exit 1
fi

