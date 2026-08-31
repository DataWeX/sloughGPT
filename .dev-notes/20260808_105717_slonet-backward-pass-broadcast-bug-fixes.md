---
id: 20260808_105717_slonet-backward-pass-broadcast-bug-fixes
title: SloNet backward-pass broadcast bug fixes
status: done
tags: training,slonet
created: 2026-08-08T10:57:17.153732+00:00
---

SloNet backward-pass broadcast bug fixes

Roadmap #1 investigation: the SloNet backward-pass broadcast failures cited in ROADMAP.md are already resolved (test_tokenizer.py, test_slonet_broadcast.py, test_slonet_bidirectional_dag.py all pass). Full training/slonet/tokenizer sweep surfaced one remaining bug: SloNet.get_user_adapter() hardcoded the npz load path to <repo_root>/data/user_adapters while the two legacy tests wrote CWD-relative data/user_adapters, so persisted adapters were never reloaded (silently fell back to fresh random weights). Fixed by adding a data_dir override param (defaults to repo-root /data, matching PerUserLORAStore.store_path convention) and updated test_load_from_npz + test_shape_mismatch_keeps_fresh to pass a hermetic tmp_path. Verified: full slonet+training+tokenizer+feedback/lora/workflow sweep green, py_compile OK.