# HDT Training Configurations

This directory contains SLURM batch scripts for training HDT models on a cluster.

## Directory Structure

```
training_configs/
├── hyperparams.json      # Training hyperparameters
├── load_hyperparams.sh   # Helper script to load hyperparameters
├── train_cluster.sh      # SLURM batch training script
├── train_local.sh        # Local GPU training script
└── README.md             # This file
```

## Usage

### Submit a Cluster Training Job

```bash
cd /home/ka/ka_stud/ka_upygb/repos/lhdt
sbatch training_configs/train_cluster.sh
```

### Run Training on Local GPU

For development or testing on a single machine:

```bash
cd /home/ka/ka_stud/ka_upygb/repos/lhdt
./training_configs/train_local.sh
```

### Monitor Job Status

```bash
# Check job status
squeue -u $USER

# View live training log (replace JOBID)
tail -f logs/training_hdt_JOBID.out

# View error log (replace JOBID)
tail -f logs/training_hdt_JOBID.err
```

### Cancel a Job

```bash
scancel JOBID
```

## Configuration

### Modifying Training Parameters

All training hyperparameters are stored in `hyperparams.json`. Edit this file to adjust training:

```json
{
  "save_dir": "logs/hdt_ed_pretrain",
  "tok_name": "google-t5/t5-base",
  "num_encoder_layers": 6,
  "num_decoder_layers": 6,
  "max_input_length": 8192,
  "max_output_length": 512,
  "mlm_probability": 0.15,
  "learning_rate": 0.001,
  "batch_size": 4,
  "accumulate_grad_batches": 16,
  "budget": 48,
  "num_gpus": 4,
  "cache_dir": "cache",
  "encoder_only": false
}
```

Both `train_cluster.sh` and `train_local.sh` read from this file, ensuring consistent configuration.

### SLURM Resource Configuration

Adjust SLURM settings in the script header:

```bash
#SBATCH --partition=gpu_h100_il     # GPU partition
#SBATCH --mem=510000mb               # Total memory
#SBATCH --time=48:00:00              # Maximum walltime
#SBATCH --gres=gpu:4                 # Number of GPUs
#SBATCH --cpus-per-task=24           # CPUs per GPU
```

## Training Configuration

### Default Configuration

- **Purpose**: Pre-train HDT-ED (encoder-decoder) on long documents
- **Resources**: 4x H100 GPUs, 48 hours
- **Model**: 6-layer encoder + 6-layer decoder (~112M params)
- **Context**: 8192 input tokens, 512 output tokens
- **Dataset**: ArXiv + Wikipedia + HUPD (auto-downloaded)

### Creating Custom Configurations

To create a variant with different parameters:

```bash
# Copy the base script
cp training_configs/train_cluster.sh training_configs/train_custom.sh

# Edit parameters
vim training_configs/train_custom.sh

# Submit
sbatch training_configs/train_custom.sh
```

## Output

After training completes:

```
logs/hdt_ed_pretrain_YYYYMMDD_HHMMSS/
├── config.json                # Model configuration
├── pytorch_model.bin          # Model weights
├── tokenizer_config.json      # Tokenizer config
├── tokenizer.json             # Tokenizer vocabulary
└── special_tokens_map.json    # Special tokens
```

## Using Trained Model

Load your trained model:

```python
from src.HDT import HDTForConditionalGeneration, HDTConfig, HDTTokenizer
from transformers import AutoTokenizer

model_path = "logs/hdt_ed_pretrain_YYYYMMDD_HHMMSS"

# Load config and model
config = HDTConfig.from_pretrained(model_path)
model = HDTForConditionalGeneration.from_pretrained(model_path, config=config)

# Load tokenizers
base_tokenizer = AutoTokenizer.from_pretrained(model_path)
hdt_tokenizer = HDTTokenizer(base_tokenizer, max_document_length=8192)
```

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE`
- Increase `ACCUMULATE_GRAD_BATCHES` to maintain effective batch size
- Reduce `MAX_INPUT_LENGTH`

### Slow Training
- Increase `BATCH_SIZE` if memory allows
- Decrease `ACCUMULATE_GRAD_BATCHES`
- Use fewer/smaller layers

### Dataset Issues
- Datasets are auto-downloaded to `cache/` directory
- Ensure internet connectivity on compute nodes
- Pre-download datasets if needed:
  ```python
  from datasets import load_dataset
  load_dataset('howey/unarXive', cache_dir='cache')
  load_dataset('howey/wiki_en', cache_dir='cache')
  load_dataset('howey/hupd', cache_dir='cache')
  ```

## Notes

- Training on 4x H100 GPUs for 48 hours processes ~2.6B tokens
- Model checkpoints are saved periodically during training
- Final model is saved to the `--save_dir` location
- Use the trained model with the chat script or fine-tune on downstream tasks

