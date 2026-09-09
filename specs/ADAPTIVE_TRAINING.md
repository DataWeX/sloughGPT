# Adaptive Training Infrastructure — Engineering Spec

## Overview

The adaptive training infrastructure learns from past training runs to recommend better configurations over time. It replaces static lookup tables with a feedback loop: record outcomes → find patterns → recommend better configs → record new outcomes.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User / Auto-Config                    │
│  "Train on this dataset" → auto_configure()             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              AdaptiveConfigEngine                        │
│  1. Query history for similar runs                      │
│  2. Weighted average of best configs                    │
│  3. Explore when confidence is low                      │
│  4. Return AdaptiveRecommendation                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│             TrainingOutcomeTracker                       │
│  - Records config + results as JSONL                    │
│  - Queries by dataset/model/method                      │
│  - Computes quality scores                              │
│  - File: ~/.config/sloughgpt/training_history.jsonl     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              PersistentSettings                          │
│  - User-facing config (generation, voice, training)     │
│  - Survives restarts (JSON file)                        │
│  - Change listeners for reactive updates                │
│  - File: ~/.config/sloughgpt/settings.json              │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. TrainingOutcomeTracker (`outcome_tracker.py`)

Records every training run's config and results.

```python
outcome = TrainingOutcome(
    run_id="run_123",
    dataset="my_data",
    dataset_size=5000,
    model="gpt2",
    method="finetune",
    epochs=5,
    batch_size=8,
    learning_rate=2e-4,
    final_loss=2.4,
    converged=True,
)
tracker.record(outcome)
```

**Quality score** (0-1) is computed from:
- Final loss (50% weight — lower is better)
- Convergence (30% weight — did it converge?)
- Perplexity (20% weight — lower is better)
- Early stopping penalty (-10% — wasted compute)

**Storage format:** JSONL (one JSON object per line)
- Append-only for efficiency
- No locking needed (single-writer)
- Human-readable for debugging

### 2. AdaptiveConfigEngine (`adaptive_config.py`)

Learns from history to recommend configs.

**Algorithm:**
1. Find runs with same (dataset_size_category, model, method)
2. Weight by quality score (better runs influence more)
3. Compute weighted average of hyperparameters
4. Confidence = f(num_runs, consistency_of_results)
5. If confidence < 0.8, suggest variations for exploration

**Fallback:** When no history exists, uses the same heuristics as `auto_config.py`.

**Integration:** `auto_configure()` calls the adaptive engine after building the static config. If the engine has enough data (≥3 runs, confidence > 0.5), it overrides the static values.

### 3. PersistentSettings (`domains/settings/persistent.py`)

User-facing configuration that survives restarts.

**Sections:**
- `generation` — temperature, top_p, top_k, repetition_penalty, max_new_tokens
- `training` — preferred_model, auto_train, method, max_checkpoints
- `adaptive` — enabled, exploration_rate, learning_enabled
- `voice` — noise_gate_db, target_level_db, vad settings, agc
- `ui` — theme, language, compact_mode

**API:**
```python
settings = get_settings()
settings.update("generation", temperature=0.5)
settings.get("voice", "noise_gate_db")
settings.reset()
settings.on_change(lambda s: print("Settings changed"))
```

## Data Flow

### Training Run Lifecycle

1. User triggers training (or auto-train fires)
2. `auto_configure()` builds static config from dataset analysis
3. Adaptive engine checks history for similar runs
4. If history exists → override config with learned values
5. Training runs, produces outcome (loss, convergence, time)
6. `update_after_run(outcome)` records to history
7. Next run benefits from this outcome

### Settings Lifecycle

1. User adjusts settings via API or UI
2. `PersistentSettings.update()` writes to disk immediately
3. Change listeners notify dependent systems
4. On restart, settings are loaded from disk
5. No data loss, no reset on server restart

## File Locations

| File | Purpose |
|------|---------|
| `~/.config/sloughgpt/training_history.jsonl` | Training outcomes |
| `~/.config/sloughgpt/settings.json` | User settings |

## Testing

All components are tested in `tests/test_adaptive_infra.py`:
- Outcome tracker: record, load, query, stats, quality scoring
- Adaptive engine: recommend, fallback, exploration, insights
- Persistent settings: update, persist, load, reset, listeners
- Integration: auto_configure with adaptive override
