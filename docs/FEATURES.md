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
| List datasets | `python3 cli.py datasets list` | ✅ Done |
| Dataset stats | `python3 cli.py datasets stats <name>` | ✅ Done |
| Search HuggingFace | `python3 cli.py datasets search <query>` | ✅ Done |
| Search GitHub | `python3 cli.py datasets search <query> --source github` | ✅ Done |
| Search Books | `python3 api.py datasets/search/books?query=<title>` | ✅ Done |
| Export dataset | `python3 cli.py datasets export <name>` | ✅ Done |
| Import GitHub | `python3 cli.py datasets github <url> [name]` | ✅ Done |
| Import HuggingFace | `python3 cli.py datasets hf <dataset_id> [name]` | ✅ Done |
| Import URL | `python3 cli.py datasets url <url> <name>` | ✅ Done |
| Data stats | `python3 cli.py data stats <path>` | ✅ Done |
| Data validate | `python3 cli.py data validate <path>` | ✅ Done |
| Train model | `python3 cli.py train --dataset <name>` | ✅ Done |
| Multi-dataset | `python3 cli.py train --datasets shakespeare,code` | ✅ Done |
| Dataset ratios | `python3 cli.py train --datasets shakespeare,code --ratios 0.7,0.3` | ✅ Done |
| Feedback export | `python3 cli.py feedback-export -o data.jsonl` | ✅ Done |
| Auto-train | `python3 cli.py autotrain start stop status` | ✅ Done |
| Model presets | `python3 cli.py train --preset small medium large` | ✅ Done |

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