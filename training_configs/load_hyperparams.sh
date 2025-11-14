#!/bin/bash

# Function to load hyperparameters from JSON file
load_hyperparams() {
    local json_file="$1"
    
    if [ ! -f "$json_file" ]; then
        echo "Error: Hyperparameter file not found: $json_file"
        exit 1
    fi
    
    echo "Loading hyperparameters from: $json_file"
    
    # Use jq to parse JSON if available, otherwise use python
    if command -v jq &> /dev/null; then
        # Load parameters using jq
        export SAVE_DIR=$(jq -r '.save_dir // empty' "$json_file")
        export TOK_NAME=$(jq -r '.tok_name // empty' "$json_file")
        export NUM_ENCODER_LAYERS=$(jq -r '.num_encoder_layers // empty' "$json_file")
        export NUM_DECODER_LAYERS=$(jq -r '.num_decoder_layers // empty' "$json_file")
        export MAX_INPUT_LENGTH=$(jq -r '.max_input_length // empty' "$json_file")
        export MAX_OUTPUT_LENGTH=$(jq -r '.max_output_length // empty' "$json_file")
        export MLM_PROBABILITY=$(jq -r '.mlm_probability // empty' "$json_file")
        export LEARNING_RATE=$(jq -r '.learning_rate // empty' "$json_file")
        export BATCH_SIZE=$(jq -r '.batch_size // empty' "$json_file")
        export ACCUMULATE_GRAD_BATCHES=$(jq -r '.accumulate_grad_batches // empty' "$json_file")
        export BUDGET=$(jq -r '.budget // empty' "$json_file")
        export NUM_GPUS=$(jq -r '.num_gpus // empty' "$json_file")
        export CACHE_DIR=$(jq -r '.cache_dir // empty' "$json_file")
        export ENCODER_ONLY=$(jq -r '.encoder_only // "false"' "$json_file")
        export WANDB_ENABLED=$(jq -r '.wandb_enabled // "false"' "$json_file")
        export GRADIENT_CHECKPOINTING=$(jq -r '.gradient_checkpointing // "false"' "$json_file")
    else
        # Fallback to python if jq is not available
        export SAVE_DIR=$(python -c "import json; print(json.load(open('$json_file')).get('save_dir', ''))")
        export TOK_NAME=$(python -c "import json; print(json.load(open('$json_file')).get('tok_name', ''))")
        export NUM_ENCODER_LAYERS=$(python -c "import json; print(json.load(open('$json_file')).get('num_encoder_layers', ''))")
        export NUM_DECODER_LAYERS=$(python -c "import json; print(json.load(open('$json_file')).get('num_decoder_layers', ''))")
        export MAX_INPUT_LENGTH=$(python -c "import json; print(json.load(open('$json_file')).get('max_input_length', ''))")
        export MAX_OUTPUT_LENGTH=$(python -c "import json; print(json.load(open('$json_file')).get('max_output_length', ''))")
        export MLM_PROBABILITY=$(python -c "import json; print(json.load(open('$json_file')).get('mlm_probability', ''))")
        export LEARNING_RATE=$(python -c "import json; print(json.load(open('$json_file')).get('learning_rate', ''))")
        export BATCH_SIZE=$(python -c "import json; print(json.load(open('$json_file')).get('batch_size', ''))")
        export ACCUMULATE_GRAD_BATCHES=$(python -c "import json; print(json.load(open('$json_file')).get('accumulate_grad_batches', ''))")
        export BUDGET=$(python -c "import json; print(json.load(open('$json_file')).get('budget', ''))")
        export NUM_GPUS=$(python -c "import json; print(json.load(open('$json_file')).get('num_gpus', ''))")
        export CACHE_DIR=$(python -c "import json; print(json.load(open('$json_file')).get('cache_dir', ''))")
        export ENCODER_ONLY=$(python -c "import json; print(str(json.load(open('$json_file')).get('encoder_only', False)).lower())")
        export WANDB_ENABLED=$(python -c "import json; print(str(json.load(open('$json_file')).get('wandb_enabled', False)).lower())")
        export GRADIENT_CHECKPOINTING=$(python -c "import json; print(str(json.load(open('$json_file')).get('gradient_checkpointing', False)).lower())")
    fi
    
    echo "Hyperparameters loaded successfully"
}

