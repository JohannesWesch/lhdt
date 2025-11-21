def _set_args(learning_rate, B, T):
    global lr, batch_size, budget
    lr = learning_rate
    batch_size = B
    budget = T


num_labels = 1
# Paper uses 1e-3 learning rate for CRAMMING-style pretraining
lr = 1e-3
optimizer_name = "adamw"
# AdamW hyperparameters matching the paper
optimizer_hyperparams = {"weight_decay": 0.01, "eps": 1e-8}
# Small batch size per GPU with gradient accumulation (paper: 1 GPU, 24h budget)
batch_size = 2
seed = 123
scheduler_name = "cosine_warmup"
scheduler_frequency = 1
scheduler_interval = "step"
scheduler_monitor = "train_loss"
# 10% warmup for inverse sqrt schedule (CRAMMING style)
scheduer_cut_frac = 0.1
budget = 24  # hours - academic budget as per paper
num_gpus = 1
task_type = "multiclass"  # Or multilabel, only useful for classification tasks