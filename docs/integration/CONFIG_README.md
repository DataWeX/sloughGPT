# SLO Configuration System

**Status:** This document describes a **planned** `.config` format. This repository does **not** currently ship a `train.py` driver or `config/*.config` files at the repo root. For working training flows today, use **`train_sloughgpt.py`**, **`domains.training.train_pipeline`**, or **`cli.py train`** (see **QUICKSTART.md**).

**YAML today:** the repo ships **`config.yaml`** plus **`config_loader.py`** (**`load_config`**, **`merge_args_with_config`**). Local **`cli.py train`** loads that file, applies CLI overrides, resolves device with **`get_device`**, then builds **`SloughGPTTrainer`**; see **`apps/cli/README.md`** for flags and fields.

This system is intended to provide a Dockerfile/Modelfile-style configuration format for SLO training.

## Format

Config files use a simple, declarative format:

```
# Comments start with #
FROM base_config  # Inherit from another config (optional)

MODEL n_layer=6 n_head=6 n_embd=384 dropout=0.1
MODEL use_rope=True use_rmsnorm=True use_swiglu=False

DATA dataset=mydata batch_size=8 block_size=512
DATA gradient_accumulation_steps=1

TRAIN learning_rate=1e-3 max_iters=3000 weight_decay=0.1
TRAIN beta1=0.9 beta2=0.95 grad_clip=1.0

LR_SCHEDULE decay_lr=True warmup_iters=100 lr_decay_iters=3000
LR_SCHEDULE min_lr=1e-4

LOGGING out_dir=out-mydata eval_interval=250 log_interval=10
LOGGING eval_iters=50 always_save_checkpoint=True wandb_log=False

SYSTEM device=cuda dtype=bfloat16 compile=True
DDP backend=nccl
```

## Directives

- `FROM` - Inherit from base config file
- `MODEL` - Model architecture parameters
- `DATA` - Dataset and batching parameters  
- `TRAIN` - Training hyperparameters
- `LR_SCHEDULE` - Learning rate scheduling
- `SYSTEM` - Device and compilation settings
- `LOGGING` - Output and monitoring settings
- `DDP` - Distributed training settings
- `ADVANCED` - Advanced model features
- `ENV` - Environment variables

## Usage (planned)

```bash
# Intended interface (not present in the repo yet)
python3 train.py config/small.config
python3 train.py config/standard.config --batch_size=16
python3 train.py config/large.config
python3 train.py config/train_mydata.py
```

## Inheritance

Config files can inherit from base configs:

```
FROM standard.config

# Override specific parameters
MODEL n_layer=8 n_embd=512
DATA batch_size=16
LOGGING wandb_run_name=custom
```

## Available Configs

- `config/small.config` - Tiny model for testing (2 layers, 128 dim)
- `config/standard.config` - Medium model (6 layers, 384 dim) 
- `config/large.config` - Large model for serious training (12 layers, 768 dim)
- `config/inheritance-test.config` - Example of inheritance

## Validation

The system validates parameters automatically:
- Type checking (int, float, bool, str)
- Range validation (e.g., 0 <= dropout <= 1)
- Architecture constraints (e.g., n_embd % n_head == 0)

## Benefits

- **Clean**: No Python syntax, just simple key=value pairs
- **Declarative**: Focus on what to configure, not how
- **Inheritance**: Reuse and extend base configs
- **Validation**: Automatic parameter checking
- **Compatible**: Works alongside existing Python configs