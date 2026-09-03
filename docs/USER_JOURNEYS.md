# User Journeys

End-to-end user flows tested via Playwright in `packages/core-py/tests/test_user_journeys.py`.

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
| Old Route | Redirects To |
|-----------|-------------|
| `/companion` | `/souls` |
| `/evaluate` | `/benchmark` |
| `/memory` | `/knowledge` |
| `/collections` | `/datasets` |
| `/self-train` | `/training` |
| `/admin` | `/settings` |
| `/images` | `/files` |
| `/session` | `/shell` |

## Untested Routes

The following routes have `page.tsx` files but are not covered by Playwright tests:

**Core**: `/training/job/[id]`, `/dataset/[id]`, `/model/[id]`
**Agents/Souls**: `/agents`, `/souls`
**Tools**: `/tokenizer`, `/benchmark`, `/compare`, `/shell`, `/errors`, `/monitoring`
**Supporting**: `/feedback`, `/files`, `/security`, `/developer`, `/magazine`, `/learn`
**Advanced**: `/lora-eval`, `/registry`, `/meta-weights`, `/voice`, `/session`, `/companion`, `/self-train`, `/kb`, `/docstore`, `/multimodal`, `/collections`, `/infer`, `/auto-train`, `/vector`, `/token-tree`, `/experiments`, `/workflow`, `/world`, `/vm`, `/kanban`, `/auth`, `/rate-limit`, `/export`, `/memory`

## Running Tests

```bash
# Full suite
.venv/bin/python -m pytest packages/core-py/tests/test_user_journeys.py -x -v

# Single journey class
.venv/bin/python -m pytest packages/core-py/tests/test_user_journeys.py::TestDatasetsImport -v
```

Results are saved to `tests/test_results/user_journey_results.json`.
