# Dataset Features

## Import Sources

| Source | Endpoint | Status | Notes |
|--------|-----------|--------|-------|
| GitHub | `/datasets/import/github` | ✅ Done | Clones repo via git |
| HuggingFace | `/datasets/import/huggingface` | ✅ Done | Uses HF Hub SDK |
| URL | `/datasets/import/url` | ✅ Done | HTTP fetch |
| Kaggle | `/datasets/import/kaggle` | ✅ Done | Uses kaggle CLI |
| CSV | `/datasets/import/csv` | ✅ Done | Converts CSV to JSONL |
| Local | `/datasets/import/local` | ✅ Done | File picker |

## Backend Features

| Feature | Status | Notes |
|---------|--------|-------|
| List datasets | ✅ Done | `/datasets` |
| Search datasets | ✅ Done | `?q=query` filter |
| Download dataset | ✅ Done | `/datasets/{id}/download` |
| Preview dataset | ✅ Done | `/datasets/{id}/preview` |
| Delete dataset | ✅ Done | `/datasets/{id}` DELETE |
| Edit metadata | ✅ Done | `/datasets/{id}` PATCH |
| Versions | ✅ Done | `/datasets/{id}/versions` |
| Validate | ✅ Done | `/datasets/{id}/validate` |

## Frontend Features (Web UI)

| Feature | Status | Notes |
|---------|--------|-------|
| Dataset list view | ✅ Done | Cards/table |
| Import modal | ✅ Done | Multi-source |
| Preview modal | ✅ Done | Sample data |
| Delete confirmation | ✅ Done | AlertDialog |
| Edit metadata | ✅ Done | Dialog |
| Search/filter | ✅ Done | |
| Versions history | ✅ Done | |

## Missing / To Build

| Feature | Priority | Status |
|---------|----------|--------|
| Online search (HF) | High | ✅ Done |
| Online search (GitHub) | High | ✅ Done |
| Books by ISBN | High | ✅ Done |
| Dataset statistics/stats | Medium | ✅ Done |
| Quick train workflow | Medium | ✅ Done ( workflow) |
| Drag & drop upload | Medium | ✅ Done |
| Dataset validation UI | Low | ✅ CLI Done |
| Export dataset | Low | ✅ Done |

## CLI Tools (cli.py)

| Tool | Command | Status |
|------|---------|--------|
| List datasets | `./sloughgpt datasets list` | ✅ Done |
| Dataset stats | `./sloughgpt datasets stats <name>` | ✅ Done |
| Search HuggingFace | `./sloughgpt datasets search <query>` | ✅ Done |
| Search GitHub | `./sloughgpt datasets search <query> --source github` | ✅ Done |
| Search Books | `python3 api.py datasets/search/books?query=<title>` | ✅ Done |
| Export dataset | `./sloughgpt datasets export <name>` | ✅ Done |
| Import GitHub | `./sloughgpt datasets github <url> [name]` | ✅ Done |
| Import HuggingFace | `./sloughgpt datasets hf <dataset_id> [name]` | ✅ Done |
| Import URL | `./sloughgpt datasets url <url> <name>` | ✅ Done |
| Data stats | `./sloughgpt data stats <path>` | ✅ Done |
| Data validate | `./sloughgpt data validate <path>` | ✅ Done |
| Train model | `./sloughgpt train --dataset <name>` | ✅ Done |
| Multi-dataset | `./sloughgpt train --datasets shakespeare,code` | ✅ Done |
| Dataset ratios | `./sloughgpt train --datasets shakespeare,code --ratios 0.7,0.3` | ✅ Done |
| Feedback export | `./sloughgpt feedback-export -o data.jsonl` | ✅ Done |
| Auto-train | `./sloughgpt autotrain start stop status` | ✅ Done |
| Model presets | `./sloughgpt train --preset small medium large` | ✅ Done |
| Native SloNet training | `sloughgpt train native --dataset <file> --steps N` | ✅ Done |

## Native SloNet Training (torch-free)

The `train native` command trains a SloNet model from scratch on pure numpy — no
PyTorch, no HuggingFace weights. Checkpoints are saved as `.soul` and load via
`SloNetChatProvider.from_soul()`.

```bash
PYTHONPATH=apps/cli/src python3 -m cli train native \
  --dataset datasets/tinyshakespeare/input.txt \
  --steps 2500 --embed 64 --layers 2 --heads 4 --block 128 \
  --batch 16 --lr 3e-3 --checkpoint-dir models/slonet-native \
  --checkpoint-interval 500 --eval-interval 250 \
  --soul-name sloughgpt-native
```

Key flags: `--steps`, `--embed/--layers/--heads/--block` (arch), `--batch`,
`--lr`, `--weight-decay`, `--scheduler`, `--warmup`, `--min-lr`, `--grad-norm`,
`--checkpoint-dir`, `--checkpoint-interval`, `--max-checkpoints`,
`--eval-interval`, `--log-interval`, `--soul-name`, `--save-stem`,
`--save-format` (`sou`/`npz`), `--resume PATH`, `--resume-latest`.

### Checkpoint retention

Checkpoints never accumulate. During training only the `--max-checkpoints`
newest `.soul` files are kept (so a crashed run can resume via
`--resume-latest`); on completion the trainer writes the final checkpoint and
the CLI removes all intermediate files, leaving **one** model file
(`<checkpoint-dir>/<soul-name>.soul` plus its `.meta.json` sidecar).

### Live progress bar

`train native`, `train`, `train embed`, and `distill` render a live
`TrainingProgressBar` (`apps/cli/src/utils/training_progress.py`). `train
native` and `train` feed it the trainer's `on_progress` dicts directly;
`train embed` and `distill` adapt their epoch/step callbacks into the same
dict shape. It shows step/total, epoch, train loss with a recent
loss sparkline, eval loss (best-tracked), learning rate, throughput (it/s),
ETA, and elapsed time. On a TTY it updates one line in place; piped/redirected
output prints one complete line per update (log-friendly). Total steps are
taken from `--steps` or inferred from `epochs × steps_per_epoch` (auto-corrected
when `--max-steps` caps the run).

## Quick Train Workflow

```bash
# Single dataset
python3 apps/cli/cli.py train --dataset shakespeare --epochs 3

# Multiple datasets with equal weighting
python3 apps/cli/cli.py train --datasets shakespeare,code --epochs 3

# Multiple datasets with custom ratios (70% shakespeare, 30% code)
python3 apps/cli/cli.py train --datasets shakespeare,code --ratios 0.7,0.3 --epochs 3
```

## Data Types

| Type | Format | Location |
|------|--------|----------|
| text | `input.txt` | Plain text file |
| corpus | `corpus.jsonl` | Structured JSON Lines |
