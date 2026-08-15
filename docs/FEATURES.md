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
| Token-tree tokenizer | `sloughgpt train native --tokenizer token-tree --token-vocab-size N` | ✅ Done |

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
  `--save-format` (DEPRECATED — ignored; checkpoints are always `.soul`), `--resume PATH`, `--resume-latest`.

### Token-tree tokenizer

By default `train native` trains on raw characters (one id per char). Pass
`--tokenizer token-tree` to first learn a BPE token tree over the corpus
(`--token-vocab-size`, default 512) and train the model on the subword tokens
instead:

```bash
PYTHONPATH=apps/cli/src python3 -m cli train native \
  --dataset datasets/tinyshakespeare/input.txt \
  --tokenizer token-tree --token-vocab-size 512 \
  --embed 64 --layers 2 --heads 4 --block 128 --batch 16 --lr 3e-3 \
  --soul-name sloughgpt-bpe
```

The trained tree is embedded in the `.soul` metadata (`tokenizer.token_tree`
via `TokenTree.to_dict()`), so `SloNetChatProvider.from_soul()` reconstructs a
`_TreeTokenizer` and reproduces the exact same BPE encoding at inference time —
no external tokenizer file needed.

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

## Embedder Quality Gate

`sloughgpt train embed` records a quality gate in the checkpoint metadata,
computed from the real training corpus (no hardcoded pairs). Three metrics are
stored for a deterministic sample of probe texts:

| Metric | Meaning |
|--------|---------|
| `degenerate_fraction` | fraction of probe pairs whose cosine is ~1.0 (identical) |
| `mean_cosine` | mean off-diagonal probe cosine — collapse detector |
| `nn_agreement` | mean top-3 neighbour overlap with the n-gram reference (diagnostic) |

### Anisotropy debiasing

SloNet encoders collapse toward a common direction (raw mean cosine ≈ 0.93+ for
small corpora). At save time the corpus mean embedding is computed from the
probe texts and stored in the checkpoint (`embed_mean`). At inference every
embedding is mean-subtracted and re-normalized — the standard BERT-whitening
debias — which re-centers the space (mean cosine ≈ 0.0) and recovers the
discriminative residuals. The quality metrics are computed on this deployed,
debiased space.

At load time `simple_embed` adopts the trained embedder for vector search only
if `acceptable()` passes: at least 2 probes, `degenerate_fraction < 0.25` and
`mean_cosine < 0.90`. A checkpoint that collapses even after debiasing is
rejected and vector search falls back to the zero-download n-gram TF-IDF
embedder. Checkpoints trained before the gate existed carry no quality metadata
and are also rejected — retrain to record it. The CLI prints the verdict and the
three metrics after training.

## Data Types

| Type | Format | Location |
|------|--------|----------|
| text | `input.txt` | Plain text file |
| corpus | `corpus.jsonl` | Structured JSON Lines |
