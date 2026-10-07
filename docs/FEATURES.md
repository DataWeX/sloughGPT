# Dataset Features

## Import Sources

| Source      | Endpoint                       | Status  | Notes                 |
| ----------- | ------------------------------ | ------- | --------------------- |
| GitHub      | `/datasets/import/github`      | ✅ Done | Clones repo via git   |
| HuggingFace | `/datasets/import/huggingface` | ✅ Done | Uses HF Hub SDK       |
| URL         | `/datasets/import/url`         | ✅ Done | HTTP fetch            |
| Kaggle      | `/datasets/import/kaggle`      | ✅ Done | Uses kaggle CLI       |
| CSV         | `/datasets/import/csv`         | ✅ Done | Converts CSV to JSONL |
| Local       | `/datasets/import/local`       | ✅ Done | File picker           |

## Backend Features

| Feature          | Status  | Notes                     |
| ---------------- | ------- | ------------------------- |
| List datasets    | ✅ Done | `/datasets`               |
| Search datasets  | ✅ Done | `?q=query` filter         |
| Download dataset | ✅ Done | `/datasets/{id}/download` |
| Preview dataset  | ✅ Done | `/datasets/{id}/preview`  |
| Delete dataset   | ✅ Done | `/datasets/{id}` DELETE   |
| Edit metadata    | ✅ Done | `/datasets/{id}` PATCH    |
| Versions         | ✅ Done | `/datasets/{id}/versions` |
| Validate         | ✅ Done | `/datasets/{id}/validate` |

## Frontend Features (Web UI)

| Feature             | Status  | Notes        |
| ------------------- | ------- | ------------ |
| Dataset list view   | ✅ Done | Cards/table  |
| Import modal        | ✅ Done | Multi-source |
| Preview modal       | ✅ Done | Sample data  |
| Delete confirmation | ✅ Done | AlertDialog  |
| Edit metadata       | ✅ Done | Dialog       |
| Search/filter       | ✅ Done |              |
| Versions history    | ✅ Done |              |

## Missing / To Build

| Feature                  | Priority | Status              |
| ------------------------ | -------- | ------------------- |
| Online search (HF)       | High     | ✅ Done             |
| Online search (GitHub)   | High     | ✅ Done             |
| Books by ISBN            | High     | ✅ Done             |
| Dataset statistics/stats | Medium   | ✅ Done             |
| Quick train workflow     | Medium   | ✅ Done ( workflow) |
| Drag & drop upload       | Medium   | ✅ Done             |
| Dataset validation UI    | Low      | ✅ CLI Done         |
| Export dataset           | Low      | ✅ Done             |

## CLI Tools (cli.py)

| Tool                   | Command                                                              | Status  |
| ---------------------- | -------------------------------------------------------------------- | ------- |
| List datasets          | `./sloughgpt datasets list`                                          | ✅ Done |
| Dataset stats          | `./sloughgpt datasets stats <name>`                                  | ✅ Done |
| Search HuggingFace     | `./sloughgpt datasets search <query>`                                | ✅ Done |
| Search GitHub          | `./sloughgpt datasets search <query> --source github`                | ✅ Done |
| Search Books           | `python3 api.py datasets/search/books?query=<title>`                 | ✅ Done |
| Export dataset         | `./sloughgpt datasets export <name>`                                 | ✅ Done |
| Import GitHub          | `./sloughgpt datasets github <url> [name]`                           | ✅ Done |
| Import HuggingFace     | `./sloughgpt datasets hf <dataset_id> [name]`                        | ✅ Done |
| Import URL             | `./sloughgpt datasets url <url> <name>`                              | ✅ Done |
| Data stats             | `./sloughgpt data stats <path>`                                      | ✅ Done |
| Data validate          | `./sloughgpt data validate <path>`                                   | ✅ Done |
| Train model            | `./sloughgpt train --dataset <name>`                                 | ✅ Done |
| Multi-dataset          | `./sloughgpt train --datasets shakespeare,code`                      | ✅ Done |
| Dataset ratios         | `./sloughgpt train --datasets shakespeare,code --ratios 0.7,0.3`     | ✅ Done |
| Feedback export        | `./sloughgpt feedback-export -o data.jsonl`                          | ✅ Done |
| Auto-train             | `./sloughgpt autotrain start stop status`                            | ✅ Done |
| Model presets          | `./sloughgpt train --preset small medium large`                      | ✅ Done |
| Native SloNet training | `sloughgpt train native --dataset <file> --steps N`                  | ✅ Done |
| Token-tree tokenizer   | `sloughgpt train native --tokenizer token-tree --token-vocab-size N` | ✅ Done |

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

## Model File Identity

A filename does not identify a checkpoint. Several `.soul` files share one
stem — one holds a 16:07 run at loss 4.10 while its legacy-spelled
`.soul.soul` sibling holds a 16:38 run at 3.97 — so identity is read from the
file, never from its name.

**Filename grammar.** One owner, `domain/inference/_internal/slo_format.py`.

| suffix  | role                                                |
| ------- | --------------------------------------------------- |
| `.sou`  | simple tier — aliases (collapses) to `.soul`        |
| `.soul` | canonical; the only write target                    |
| `.slo`  | interchange; always a read candidate, never written |
| `.slnc` | mmap runtime, no soul identity                      |

Write side is strict (`soul_path`), read side is tolerant
(`soul_read_candidates`): the spelling the caller _named_ is probed first and
its sibling second, because the two can be genuinely different checkpoints —
canonical-first would silently hand back the neighbour and nothing downstream
would catch it. `domain/training/_internal/checkpoints.py` probes the same
list, so there is one answer to "where is this checkpoint?" rather than two.

**Identity.** `classify_soul(path) -> SoulIdentity` derives everything from
bytes: format from the magic (a soul renamed `.bin` still loads, a `.pt`
refused by name), uniqueness from `integrity_hash`, plus `born_at`,
`final_train_loss`, every competing `variants` under the same stem, and
whether the name is `shadowed` by a second checkpoint. It never raises.

**Declared axes**, written by `save_soul(..., record=...)` and surfaced by the
`models` listing and `GET /training/checkpoints`:

| field        | comes from                          | when absent                    |
| ------------ | ----------------------------------- | ------------------------------ |
| `tier`       | the suffix, via `SOUL_TIER_POLICY`  | falls back to that policy      |
| `provenance` | `record=`, one of `SOUL_PROVENANCE` | stays unknown — never inferred |

`SOUL_PROVENANCE` is `training` / `export` / `distillation`, and `save_soul`
rejects anything else before writing: the value is stamped once and never
rewritten, so a typo would become permanent identity metadata. Tier may fall
back to the suffix because container tier really is a property of the
spelling; provenance may not, because no filename records who made a file.
Neither enters `integrity_hash`, so declaring them changes no existing hash.

**Where identity ends.** `integrity_hash` hashes the declared profile, so it
exists only where a profile does. Two sidecar shapes are in the tree:
`save_soul` writes the full ~30-key profile and stamps a hash, while older
producers write a 1–3 key descriptor (`soul_name`, occasionally `lineage` or
`soul_signature` — an architecture record, not content). The second kind
cannot yield a hash and must not be given one: synthesising a profile from
`{"soul_name": ...}` plus defaults would hand _different_ files the _same_
digest, manufacturing false "these are the same checkpoint" claims. Absence
means "not recorded", never "missing".

Detail therefore states what a file is instead of staying silent.
`checkpoint_info` fills only _missing_ identity fields from `classify_soul` —
`format` at minimum, so a stray `evil.soul` reports `not-soul` and
`traits.soul` shows its declared name disagreeing with its bytes — and never
overwrites a stored value, because read-time derivation must not drift
identity. One file per request (~7ms). The listing never classifies per row:
that is what keeps `cmd_models` at 0.20s rather than 81.74s — yet it still
shows identity, because `integrity_hash` and `provenance` are plain row
fields the scan already returns. The list renders what exists and derives
nothing, which is why it and the detail dialog agree about one file without
the list doing any of `checkpoint_info`'s work.

Search roots are declared once, in `ckpt_roots()`. The list, lookup, download
and delete carried five hand-written tuples, and `LORA_DIR` sat in the scan
but not in `load_soul`, so a checkpoint could be listed and even downloaded
yet 404 the instant it was opened (measured: 11 of 23 rows) while Delete
removed nothing. It is a function rather than a tuple so that test isolation
patching the module globals still redirects it.

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

| Metric                | Meaning                                                             |
| --------------------- | ------------------------------------------------------------------- |
| `degenerate_fraction` | fraction of probe pairs whose cosine is ~1.0 (identical)            |
| `mean_cosine`         | mean off-diagonal probe cosine — collapse detector                  |
| `nn_agreement`        | mean top-3 neighbour overlap with the n-gram reference (diagnostic) |

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

| Type   | Format         | Location              |
| ------ | -------------- | --------------------- |
| text   | `input.txt`    | Plain text file       |
| corpus | `corpus.jsonl` | Structured JSON Lines |

## Consciousness System

The consciousness system provides AI self-awareness monitoring and management with 26 pages:

### Core Pages

| Page         | Path                          | Status  | Description                     |
| ------------ | ----------------------------- | ------- | ------------------------------- |
| Dashboard    | `/consciousness/dashboard`    | ✅ Done | Main consciousness overview     |
| Health       | `/consciousness/health`       | ✅ Done | System health monitoring        |
| Training     | `/consciousness/training`     | ✅ Done | Consciousness training controls |
| Settings     | `/consciousness/settings`     | ✅ Done | Configuration management        |
| All Settings | `/consciousness/all-settings` | ✅ Done | Advanced settings               |
| Export       | `/consciousness/export`       | ✅ Done | Data export functionality       |
| History      | `/consciousness/history`      | ✅ Done | Episode history viewer          |
| Quickstart   | `/consciousness/quickstart`   | ✅ Done | Getting started guide           |

### Analysis & Debugging

| Page         | Path                          | Status  | Description                 |
| ------------ | ----------------------------- | ------- | --------------------------- |
| Compare      | `/consciousness/compare`      | ✅ Done | Configuration comparison    |
| Benchmark    | `/consciousness/benchmark`    | ✅ Done | Performance benchmarking    |
| API Explorer | `/consciousness/api-explorer` | ✅ Done | Interactive API testing     |
| Testing      | `/consciousness/testing`      | ✅ Done | Debugging and testing tools |
| Debug        | `/consciousness/debug`        | ✅ Done | Debugging tools             |
| Test Runner  | `/consciousness/test-runner`  | ✅ Done | Automated testing           |

### Documentation & Insights

| Page     | Path                      | Status  | Description         |
| -------- | ------------------------- | ------- | ------------------- |
| Docs     | `/consciousness/docs`     | ✅ Done | API documentation   |
| Help     | `/consciousness/help`     | ✅ Done | Help and FAQ        |
| Insights | `/consciousness/insights` | ✅ Done | AI-powered insights |

### Monitoring & Analytics

| Page       | Path                        | Status  | Description          |
| ---------- | --------------------------- | ------- | -------------------- |
| Alerts     | `/consciousness/alerts`     | ✅ Done | Alert management     |
| Statistics | `/consciousness/statistics` | ✅ Done | Analytics dashboard  |
| Analytics  | `/consciousness/analytics`  | ✅ Done | Trend analysis       |
| Monitor    | `/consciousness/monitor`    | ✅ Done | Real-time monitoring |
| Versions   | `/consciousness/versions`   | ✅ Done | Version history      |

### Advanced Features

| Page        | Path                         | Status  | Description                 |
| ----------- | ---------------------------- | ------- | --------------------------- |
| Playground  | `/consciousness/playground`  | ✅ Done | Interactive experimentation |
| Personality | `/consciousness/personality` | ✅ Done | Personality management      |
| Master      | `/consciousness/master`      | ✅ Done | Master dashboard            |

### Consciousness Components

| Component                 | Status  | Description          |
| ------------------------- | ------- | -------------------- |
| ConsciousnessQuickActions | ✅ Done | Quick action buttons |
| QuickActionsWrapper       | ✅ Done | Wrapper component    |
| SidebarWidget             | ✅ Done | Sidebar integration  |
| NotificationsPanel        | ✅ Done | Notification display |
| MessageBadge              | ✅ Done | Message count badge  |

### Consciousness Hooks

| Hook                          | Status  | Description             |
| ----------------------------- | ------- | ----------------------- |
| useConsciousnessBatch         | ✅ Done | Batch operations        |
| useConsciousnessLive          | ✅ Done | Live data updates       |
| useConsciousnessNotifications | ✅ Done | Notification management |
| useConsciousnessShortcuts     | ✅ Done | Keyboard shortcuts      |
| useConsciousnessStats         | ✅ Done | Statistics computation  |
| useConsciousnessStatus        | ✅ Done | Status monitoring       |

### Consciousness Lib

| Module                      | Status  | Description           |
| --------------------------- | ------- | --------------------- |
| consciousness-controller    | ✅ Done | Main controller logic |
| consciousness-bus           | ✅ Done | Event bus             |
| consciousness-notifications | ✅ Done | Notification store    |

## Tools Pages

| Tool       | Path          | Status  | Description                |
| ---------- | ------------- | ------- | -------------------------- |
| Brainstorm | `/brainstorm` | ✅ Done | AI brainstorming assistant |
| Decide     | `/decide`     | ✅ Done | Decision-making helper     |
| Explain    | `/explain`    | ✅ Done | Simple explanations        |
| Rewrite    | `/rewrite`    | ✅ Done | Text rewriting & polish    |
| Translate  | `/translate`  | ✅ Done | Text translation           |
| Wellness   | `/wellness`   | ✅ Done | Wellness & relaxation      |
| Writing    | `/writing`    | ✅ Done | Writing assistant          |

## Test Coverage

| Area                | Coverage | Tests |
| ------------------- | -------- | ----- |
| Consciousness (all) | 100%     | 384   |
| Navigation          | 100%     | 8     |
| Hooks/Lib           | 95%+     | 1793  |
| Features/Chat       | 95%+     | 1467  |
| Tools Pages         | 100%     | 7     |
