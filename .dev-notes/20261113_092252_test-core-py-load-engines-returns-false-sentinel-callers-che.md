---
id: 20261113_092252_test-core-py-load-engines-returns-false-sentinel-callers-che
status: done
board: 20261113_092252_test-core-py-load-engines-returns-false-sentinel-callers-che
title: "test-core-py: _load_engines returns False sentinel, callers check only None — 150-failure cycles cascade"
tags: ["ci", "tests", "shell", "native"]
sprint: ""
gh: null
---

# test-core-py: _load_engines returns False sentinel, callers check only None — 150-failure cycles cascade

ROOT CAUSE (diagnosed 2026-11-13 from PR #308/#312 CI logs + dependabot run 43383838040):

domain/shell/_internal/_native.py _load_engines() docstring promises 'Compile trace_core.c once (CPU + GPU) or return None', but:

1. loader failure path: except -> _ENGINES = False; return None  (benign on FIRST call)
2. cache-hit path (line ~60): if _ENGENGINES is not None: return _ENGINES  -> returns **False** once poisoned
3. success-but-unusable path: _ENGINES = eng if eng.any else False; ... return _ENGINES -> returns **False**

All three callers check only 'is None':
- domain/shell/_internal/_native.py trace() line ~179 (eng.cpu / eng.gpu -> AttributeError: 'bool' object has no attribute 'cpu')
- tests/core-py/test_cycles_native.py _native_available() line 55
- tests/core-py/test_cycles_native.py line 143 (_load_engines().cpu)

Cascade: once any loader failure/eng.any==False happens in-process, every later call crashes -> 150 failed / ~43270 passed across test_cycles* (114), test_tensor_native, test_render_neural*, test_shell_repl_more, test_vm_bus_handles, test_world_stream, test_quantization_benchmark, test_cycles_gpu + 2 collection errors. Env-dependent (loader failures vary by runner/lineage) -> byte-identical 150 on main-derived branches but success runs exist too (efad52746 had BOTH a success and a failure run).

FIX (small, contract-normalizing): make both 'return _ENGINES' sites honor the documented contract: return None when _ENGINES is False (keep False as the internal cache sentinel so we never recompile). Callers keep their 'is None' checks; optionally harden _native_available with 'if not eng'. Add a regression test: poison _ENGINES=False, call _load_engines() twice, assert both return None (second call currently returns False).

Evidence: PR #308 comment + PR #312 comment (three-strike tables, byte-identical 150/43268 across lineages); root trace domain/shell/_internal/_native.py:186 + tests/core-py/test_cycles_native.py:60.
