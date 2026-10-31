# Card 061 — One handler for all models: portable loader

- **Status**: done
- **Branch**: `feat/model-loader-portability`
- **Date**: 2026-09-26
- **Directive**: user — "(8,32) into (8,64) — we should have an all-in-one for different models, that's the first thing we should work on: portability of models using one handler."

## Trigger

`trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` (hidden=8, intermediate=32) downloads + converts to `.slnc` fine, but SloNet load fails:

```
could not broadcast input array from shape (8,32) into shape (8,64)
```

Buffer allocation derives from a different source than the weight shapes; numpy broadcast is where it explodes.

## Plan (expert-review approved)

1. **Plan-time shape verification** in `build_load_plan` + structured `ShapeMismatchError` (tensor name, expected vs actual, config dims) — no more mid-load numpy broadcast surprises.
2. **Config-first dims**: single `dims_from_config()` from config.json (hidden/intermediate/heads/kv/vocab); demote `infer_arch_from_state_dict` silent defaults (n_embed=128 etc.) to warned fallback.
3. **`alloc_params()` + handler registry** keyed by `model_type` (qwen2, gpt2): allocation only from the plan; per-arch quirks (fused gate_up, GQA repeat, tied lm_head) live in handlers.
4. **Test matrix**: parametrized loader tests across arch fixtures; tiny trl model = red-today regression test; Qwen2.5 dims must reproduce exactly (scope guard).
5. **Verify**: ruff + pytest → boot default Qwen2.5-0.5B (`SLO_BOOTSTRAP_HF=1`) → merge → resume card 062 journeys.

## Scope guard (must NOT change)

- `.slnc` on-disk format (checkpoint compatibility)
- ProcessGuard / lazy-guard serving paths
- Known-good Qwen2.5-0.5B buffer shapes (test-enforced)

## Out of scope (adjacent journey blockers, card 062)

- Router `mgr.download(model_id, total_bytes_hint)` — int passed as url (download journey suspect)
- `load_model` docstring claims safetensors→slnc conversion it never performs
- Catalog advertises models the loader cannot load


## Outcome (closed 2026-09-27)

- Landed on `feat/model-loader-portability` (`09cde017c`): plan-time shape verification + `ShapeMismatchError`, config-first `dims_from_config()`, `_assign_checked` zero-pads weight rows when config dim_ff (ceil-to-64) exceeds checkpoint rows — the exact (8,32)→(8,64) broadcast that blocked model load.
- Merged into `feat/journey-verification` (`274779188`), then to `main` (`73ab55d8f`) after merging main back in (`0a1da7248`).
- Verified: `test_loader_portability.py` 19 passed; tiny-Qwen fixture loads (`model_loaded:true` on health); ruff clean; scope guard held (no .slnc format / ProcessGuard / Qwen2.5-0.5B shape changes).
- Left as documented out-of-scope: router `mgr.download` int-as-url, `load_model` docstring conversion claim, catalog advertises unloadable models.
