# Goal 17: Track A LoRA ship

**Status:** done
**Sprint:** 2026-09-W4
**gh:** —

## Task
ROADMAP goal 17 — LoRA (`HFLoraTrainer`) trainable fine-tune: fix live path, ship quality+speed benchmark, hermetic click-through journey (dataset → start → loss → checkpoint → load into chat).

## Result
- Branch: `feat/goal17-lora-ship`
- **Live path fixes:** `lora.py` builds `HFLoraConfig` via `config_kwargs` (passes `progress_callback`, omits `target_modules` when None so default `W_q/W_k/W_v/W_o` applies); `HFLoraTrainer.train(on_progress=)` dual path; `unload_lora_adapter` import + `last_path`; `_job_summary` exposes `loss_history`/`rank`/`alpha`/`result`; `MorphTokenizer.stoi/itos` aliases
- **Regression:** consciousness `_run_training` checks model file before LoRA so missing model still returns `data_saved` (restores goal-15 contract after config became pure-spec)
- **Journey:** `tests/test_lora_ship_journey.py` — tiny slnc+tokenizer under tmp, `POST /training/lora-finetune` → poll → finite loss_history + `journey_lora.npz` → load-adapter (rank/alpha/n_params) → unload-adapter
- **Bench:** `scripts/benchmark_hf_lora.py` recorded `--kind training` — gate 61 steps, loss 4.96→4.64 (finite improving), 6.21 steps/s, adapter ~6KB
- **Tests:** 188 related green (journey + status transitions + 3 LoRA suites); ruff clean; scatter gate green
- ROADMAP 17 marked Done; kanban `20260924_033` → done

## MERGED (2026-09-24)

- Branch `feat/goal17-lora-ship` merged to `main` with `--no-ff` (`801ab140`).
- main is not pushed (do not push unless asked).
