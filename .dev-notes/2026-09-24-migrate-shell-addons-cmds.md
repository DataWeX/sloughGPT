---
id: 20260924_051_migrate_shell_addons_cmds
status: done
title: Flip plural shell addons/cmds to forward shims
tags: [architecture, migration, dual-tree]
---
# Dual-tree step 2c — plural shell addons/cmds

## Plan
1. Branch feat/migrate-shell-addons off main.
2. Convert 15 byte-identical duplicate files in packages/core-py/domains/shell/{addons,cmds}/ to forward shims (pattern: re-export from domain.shell._internal.*).
3. Gates: ruff on shell surface, shell test suites (test_shell_cmds, test_linux_cmds, test_data_cmds, test_models_cmd, test_dashboard_helpers, test_neural_*, test_module_loader, test_filesystem*), scatter, identity probe.
4. Merge to main, close card.

## Findings
- 15/15 twins byte-identical (diff verified).
- Singular tree self-contained: relative `.addons`/`.cmds` imports only.
- Zero `domains.shell` import statements anywhere; tests already retargeted.

**Completed:** merged to main `0850db392`; fixed `from pkg.__init__ import *` dual-module bug in package shims (identity mismatch); gates green; shell plural reals 15→0; total plural reals now 9.
