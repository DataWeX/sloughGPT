---
id: 20260814_081209_de-parameterize-trainer-save-validator-torch-check
title: De-parameterize trainer save() + validator torch check
status: done
tags: training,cli,validator
created: 2026-08-14T08:12:09.553173+00:00
---

De-parameterize trainer save() + validator torch check

save() always writes .soul; legacy format param kept as DEPRECATED - DeprecationWarning fired at function entry (before any side effects), typed Optional[str], value ignored. CheckpointConfig.__post_init__ DeprecationWarning for export_format not in ('', 'soul'); --save-format CLI option deprecated-but-ignored (Choice [soul,sou,npz]). config_loader save_format merge wiring removed. Validator torch check downgraded to warn. NEW apps/cli/tests/test_validator.py (14 tests) covering torch-missing-warns, CUDA/MPS passes, no-GPU warns, plus CheckResult/ValidationResult semantics and other Doctor checks. Fixed test-isolation bug (logger.disabled=True in _monotonic_time). Follow-up: _check_pytorch hardened to catch ANY torch failure, not just ImportError — broken installs raising OSError (missing libcudart.so) or missing torch.backends.mps attr now warn instead of crashing; +2 tests (test_torch_broken_install_warns_not_crashes, test_torch_without_mps_attr_warns). Verified: apps/cli 39 pass; core-py+root combined 239 pass, 2 skip, 0 warnings.