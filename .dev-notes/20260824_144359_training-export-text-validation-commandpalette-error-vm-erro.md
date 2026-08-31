---
id: 20260824_144359_training-export-text-validation-commandpalette-error-vm-erro
title: Training export-text validation, CommandPalette error, VM error sanitization
status: done
tags: backend,frontend,security
created: 2026-08-24T14:43:59.527628+00:00
---

Training export-text validation, CommandPalette error, VM error sanitization

Added Pydantic ExportTextRequest to export-text endpoint. Fixed CommandPalette model load toast. Changed vm.py to use classify_and_raise instead of leaking str(e).