---
id: 20261112_092638_ci-pyyaml-dropped-with-transformers-restore-explicit-dep-at
status: done
board: 20261112_092638_ci-pyyaml-dropped-with-transformers-restore-explicit-dep-at
title: "CI: pyyaml dropped with transformers — restore explicit dep at 5 install sites (test-root + test-server red on main)"
tags: ["ci", "bugfix"]
sprint: ""
gh: null
---

# CI: pyyaml dropped with transformers — restore explicit dep at 5 install sites (test-root + test-server red on main)

66c0da7cd (card 20261112_062436) removed transformers from 5 pip install sites; pyyaml>=6.0.0 is a declared project dep (pyproject.toml:49) but was never installed explicitly — transformers carried it transitively. Now: test-root = 6 collection errors (No module named yaml), test-server = benchmark router 500s + facade test. Fix: append pyyaml to the same 5 lines. Evidence: git show 66c0da7cd + grep pyyaml workflows (none) + 8 domain/tests files import yaml.
