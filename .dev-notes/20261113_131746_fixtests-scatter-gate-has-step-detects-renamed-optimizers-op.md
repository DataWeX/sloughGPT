---
id: 20261113_131746_fixtests-scatter-gate-has-step-detects-renamed-optimizers-op
status: done
board: 20261113_131746_fixtests-scatter-gate-has-step-detects-renamed-optimizers-op
title: "fix(tests): scatter gate — _has_step detects renamed optimizers (opt.step)"
tags: ["tests", "training", "gate"]
sprint: ""
gh: null
---

# fix(tests): scatter gate — _has_step detects renamed optimizers (opt.step)

The one-loop gate keyed on the literal optimizer.step / clip_gradients, so a renamed optimizer variable (opt.step) evaded it. Three real loops were undetected behind a green 5/5. Widened the detector, re-baselined the allowlist as pre-existing debt; ratchet still rejects loop #23.
