---
id: 20260731_223236_downcraft-resume-validated-app-autoload-integration
title: downcraft resume validated + app autoload integration
status: done
tags: downcraft,resume,infra
created: 2026-07-31T22:32:36.242134+00:00
---

downcraft resume validated + app autoload integration

Completed downcraft pause/resume validation + app autoload integration.

Fixes:
1. download_hf_model writes refs/main -> default after completion, so is_download_complete recognizes the snapshots/default layout it produces.
2. already_cached shortcut verifies disk truth via is_download_complete; stale state triggers redownload.
3. hf_home semantic fix: download_hf_model no longer coerces None -> env string. None resolves via HF_HOME + /hub (standard HF layout, matching is_download_complete and the app's safetensors_loader._get_model_dir). Explicit hf_home remains the hub root. Before, with HF_HOME set, files landed at HF_HOME/models--<id> while checks read HF_HOME/hub/models--<id> (layout mismatch).
4. apps/api/server/infrastructure/startup.py _autoload_model step 2: replaced huggingface_hub.snapshot_download with downcraft.download_hf_model (resume-aware; removed the last snapshot_download usage in the repo). Loader stack already resolves snapshots/* layout.

Tests: downcraft suite 64 passed (test_hf_hub + test_resume, incl. refs/main + stale-state tests). core-py 75 passed incl. new packages/core-py/tests/test_loader_layout_contract.py (downcraft output -> _get_model_dir + _find_safetensors + load_model_weights end-to-end via local Range server).

Live E2E demo (/tmp/opencode/dc_resume_demo.py, prajjwal1/bert-tiny): SIGKILL at 4MB/8MB mid-download -> inspect_incomplete identifies pytorch_model.bin at exact offset -> resume in a fresh process -> is_download_complete=True. PASS. Demo isolates per-run state via HF_HOME + HOME env.

Remaining integration scope: none blocking. download_manager.py already delegates to downcraft; project cache root models/hf-cache/hub already registered in downcraft.hf_hub.