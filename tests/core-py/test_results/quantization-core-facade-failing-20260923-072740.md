## quantization-core facade lane — **FAILING** (raw-verified, NOT green)
- suite family: exact per-suite glob-verbatim names below, git-ls-files-checked on disk (disk==HEAD verified at capture time)
  packages/core-py/tests/test_quant_core.py packages/core-py/tests/test_quantization.py packages/core-py/tests/test_quantized_matmul.py packages/core-py/tests/test_quantization_coverage.py 
- result: **FAILING** — exit=132 (SIGILL), raw capture line 'core dumped', partial ~17 dots at 20% (4 suites started)
- raw evidence: /tmp/quant-core-facade-batch-20260923-072642.txt (Read-verbatim: exit=132, 'timeout: the monitored command dumped core')
- status: NOT PASSED — flagged to worker (owner: quantization lane internalization); this row is the honest red record, awaiting the real quant fix, not a pass claim
- captured: 2026-09-23T07:27:40Z
