# User Journeys

End-to-end user flows driven by the **avion** computer-use library (`packages/avion`) in `tests/core-py/test_user_journeys.py` — one shared driver, so locator/backend fixes land in one place.

## Tested Journeys

### Dashboard

- **Load**: Home page (`/`) loads with substantial content (>50 chars)
- **Navigation**: Home page contains nav links to core sections

### Chat

- **Load**: Chat page (`/chat`) loads
- **Input**: Chat page has a visible textarea/input
- **Type message**: User can type a message and the value persists

### Training

- **Load**: Training page (`/training`) loads with "train" content
- **Import button**: "+ Import" button is visible
- **Tabs**: Train, Results, Settings tabs are visible
- **Train tab**: Contains training pipeline form with Start button or dataset selector
- **Results tab**: Shows training history/analytics content
- **Job detail**: `/training/job/[id]` loads with job info or not-found state
- **Job back link**: Job detail page has a back link to training list

### Datasets

- **Load**: Datasets page (`/datasets`) loads
- **Import button**: "Import" button exists
- **Import dialog**: Clicking "Import" opens a dialog
- **Kaggle option**: Radio button "Download from Kaggle" exists in dialog
- **Kaggle input**: Clicking Kaggle radio reveals `username/dataset-name` input
- **Kaggle fill**: Input can be filled with a dataset identifier
- **Kaggle import**: Full end-to-end: fill `heptapod/titanic` → click Import → "downloaded" appears (requires Kaggle CLI auth)

### Settings

- **Load**: Settings page (`/settings`) loads

### Planner

- **Load**: Planner page (`/planner`) loads

### Models

- **Load**: Models page (`/models`) loads

### Monitoring

- **Load**: Monitoring page (`/monitoring`) loads with health keywords (cpu, memory, monitor)

### Knowledge

- **Load**: Knowledge page (`/knowledge`) loads with keywords (knowledge, memory, fact)

### Redirects (Legacy Routes)

| Old Route      | Redirects To |
| -------------- | ------------ |
| `/companion`   | `/souls`     |
| `/personality` | `/souls`     |
| `/evaluate`    | `/benchmark` |
| `/memory`      | `/knowledge` |
| `/collections` | `/datasets`  |
| `/self-train`  | `/training`  |
| `/admin`       | `/settings`  |
| `/images`      | `/files`     |
| `/session`     | `/shell`     |

## Untested Routes

The following routes have `page.tsx` files but are not covered by journey tests:

**Core**: `/dataset/[id]`
**Agents/Souls**: `/agents`, `/souls`
**Tools**: `/tokenizer`, `/benchmark`, `/compare`, `/shell`, `/errors`, `/monitoring`
**Supporting**: `/feedback`, `/files`, `/security`, `/developer`, `/magazine`, `/learn`
**Advanced**: `/lora-eval`, `/registry`, `/meta-weights`, `/voice`, `/session`, `/companion`, `/self-train`, `/kb`, `/docstore`, `/multimodal`, `/collections`, `/infer`, `/auto-train`, `/vector`, `/token-tree`, `/experiments`, `/workflow`, `/world`, `/vm`, `/kanban`, `/auth`, `/rate-limit`, `/export`, `/memory`

## Running Tests

```bash
# Full suite (needs a live stack — skips itself when none is running)
scripts/python -m pytest tests/core-py/test_user_journeys.py -x -v

# Single journey class
scripts/python -m pytest tests/core-py/test_user_journeys.py::TestDatasetsImport -v
```

Results are saved to `tests/core-py/test_results/user_journey_results.json`.

## Recording Runs & Comparing Against the Previous Run

`--record` runs the same journeys but writes each run as a **run directory**
(per-step checkpoint screenshots + per-step trace), then diffs it against the
previous recorded run and prints a concise summary:

```bash
# Record a run; automatically compares vs the previous run
SLO_WEB_URL=http://localhost:5173 .venv/bin/python -m journeys --record

# Compare any two runs by hand (exit 0 = identical, 1 = changed, 2 = error)
.venv/bin/python -m avion compare <old-run-dir> <new-run-dir>
.venv/bin/python -m avion compare <old-run-dir> <new-run-dir> --json   # full diff
```

- **Where runs live**: `~/.cache/slog-journeys/runs/<run-id>/` (override with
  `SLO_JOURNEY_RUNS`). The newest **10** runs are kept; older ones are pruned
  automatically — bounded history, no silent disk growth.
- **What's in a run**: `run.json` (meta: git sha, browser, per-flow status),
  `<flow>/result.json` (step trace: name/status/duration per step),
  `<flow>/shots/NN-<step>.png` (checkpoint after every step), and — when a
  previous run existed — `comparison.json` (the raw diff; the printed summary
  is the one-line view of it).
- **What the diff flags**: step sequences (added/removed/moved/status changes),
  flow status transitions, flows added or removed, and screenshot pixel changes
  above a threshold (per-channel noise tolerance `--pixel-tol`, default 8;
  changed-pixel fraction `--changed-frac`, default 0.5%).
- **Advisory, not a gate yet**: the comparison reports; it does not fail CI.
  The `diff(1)`-style exit codes are already gate-ready for a later ratchet.
