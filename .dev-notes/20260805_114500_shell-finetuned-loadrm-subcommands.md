---
id: 20260805_114500_shell-finetuned-loadrm-subcommands
title: Shell finetuned load/rm subcommands
status: done
tags: shell,finetuned,core-py
created: 2026-08-05T11:45:00.613102+00:00
---

Shell finetuned load/rm subcommands

The shell 'finetuned' command was list-only; the API wrappers load_finetuned/delete_finetuned existed but had no command surface. Added subcommands: finetuned load <name> (POST /training/finetuned-models/{name}/load), finetuned rm|del|delete <name> (DELETE /training/finetuned-models/{name}), with usage/error handling and a 'Use:' hint after the list table. Tab completion: bare 'finetuned' completes subcommands; 'finetuned load/rm/del/delete ' completes model names via /training/finetuned-models. Help text and docs/SHELL.md updated. Tests: 7 new in test_shell_cmds.py (FakeApi gained load_finetuned/delete_finetuned) + 2 new completion tests in test_shell_repl.py; all 235 shell tests pass, py_compile clean.