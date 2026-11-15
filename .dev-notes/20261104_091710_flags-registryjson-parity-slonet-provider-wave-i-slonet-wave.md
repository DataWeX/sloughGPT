---
id: 20261104_091710_flags-registryjson-parity-slonet-provider-wave-i-slonet-wave
status: done
board: 20261104_091710_flags-registryjson-parity-slonet-provider-wave-i-slonet-wave
title: "flags: registry/JSON parity — slonet_provider_wave_i + slonet_wave_f missing from config/feature_flags.json, no gate keeps them in sync"
tags: ["tests", "flags"]
sprint: ""
gh: null
---

# flags: registry/JSON parity — slonet_provider_wave_i + slonet_wave_f missing from config/feature_flags.json, no gate keeps them in sync

Found 2026-11-04 while landing the step_trace flag (PR #262): those two flags are registered in _register_defaults() but ABSENT from config/feature_flags.json. load_config() only overlays what the JSON declares, so they silently run on code defaults and cannot be flipped via config. Pre-existing drift — not introduced by #262 (which added both code registration AND the JSON entry for step_trace).\n\nScope:\n- Add the two missing entries to config/feature_flags.json matching their code status.\n- Parity test (ratchet style): every flag in _register_defaults() must appear in config/feature_flags.json and every JSON key must be a registered flag; baseline/xfail pattern if the list should be allowed to grow deliberately.\n- Consider whether an UNKNOWN key in JSON (the other direction) should warn at load_config() time.\nGates: tests/core-py/test_feature_flags.py + Feature Tags CI.
