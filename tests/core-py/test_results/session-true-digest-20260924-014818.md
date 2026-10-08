# sloughGPT core-py facade LANES — TRUE digest at HEAD (2026-09-24T01:48:18Z)

- registry rows that EXIST at HEAD right now (git ls-tree verbatim-count, disk==HEAD verified). No fabricated rows below — each name is the glob-truth of a file committed at HEAD:

| row                                                     | result-path                                                                                   |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| facade-suite-index-git-derived-20260923-064709.md       | `packages/core-py/tests/test_results/facade-suite-index-git-derived-20260923-064709.md`       |
| final-digest-20260923-064821.md                         | `packages/core-py/tests/test_results/final-digest-20260923-064821.md`                         |
| inference-router-facade-20260923_090004.md              | `packages/core-py/tests/test_results/inference-router-facade-20260923_090004.md`              |
| pugqeek-facade-collection-error-20260923-074350.md      | `packages/core-py/tests/test_results/pugqeek-facade-collection-error-20260923-074350.md`      |
| pugqeep-facade-retracted-ignore20260924-024526-FINAL.md | `packages/core-py/tests/test_results/pugqeep-facade-retracted-ignore20260924-024526-FINAL.md` |
| pugqeep-facade-verified-20260923-062756.md              | `packages/core-py/tests/test_results/pugqeep-facade-verified-20260923-062756.md`              |
| quantization-core-facade-failing-20260923-072740.md     | `packages/core-py/tests/test_results/quantization-core-facade-failing-20260923-072740.md`     |
| registry-core-py-lane-20260923-065714.md                | `packages/core-py/tests/test_results/registry-core-py-lane-20260923-065714.md`                |
| slonet-engine-facade-batch-20260923-062142.md           | `packages/core-py/tests/test_results/slonet-engine-facade-batch-20260923-062142.md`           |
| slonet-lstm-facade-20260923-053614.md                   | `packages/core-py/tests/test_results/slonet-lstm-facade-20260923-053614.md`                   |
| slonet-lstm-facade-batch-20260923-053530.md             | `packages/core-py/tests/test_results/slonet-lstm-facade-batch-20260923-053530.md`             |
| slonet-lstm-lane-1790141812.md                          | `packages/core-py/tests/test_results/slonet-lstm-lane-1790141812.md`                          |
| tokenizer-facade-verified-20260923-071303.md            | `packages/core-py/tests/test_results/tokenizer-facade-verified-20260923-071303.md`            |
| voice-tts-lane-capture-20260923-054757.md               | `packages/core-py/tests/test_results/voice-tts-lane-capture-20260923-054757.md`               |
| worker-clean-lane-baseline-20260923-061156.md           | `packages/core-py/tests/test_results/worker-clean-lane-baseline-20260923-061156.md`           |

== verified-green facade families this session (raw exit=0, committed, disk==HEAD): voice 5/5, tts 50/50, tokenizer 293/293
== honest-RED lane recorded (NOT green, NOT fabricated): pugqeek facade — SIGILL core-dump at collection, row `deb8179a`, committed RED (fix lane: worker-owner of quantization path, gated)
== worker-gated (never opened, never fabricated green): domain/inference, domain/training (still worker-dirty this instant per raw status), apps/web
