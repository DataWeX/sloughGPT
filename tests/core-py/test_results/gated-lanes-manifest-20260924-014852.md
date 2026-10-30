# Gated lanes manifest — raw git-status probe-driven (worker-owner truth, no fabrication)

- probe: 2026-09-24T01:48:52ZZ, `git status --short -- domain/…` count at probe = 0 dirty files

| lane | owner | gate | why parked |
|------|-------|------|-----------|
| domain/inference | worker | 0 dirty (probe) | live SIGILL-bearing refactor; sealed against facade seam writes |
| domain/training | worker | $gate dirty (probe) | consolidation stage 2+ owned by worker; facade batch green already landed |
| domain/inference/* | worker | gated | LINEAR facade covered; dotted-QKV/fused-kv covered green by facade batch |
| apps/web (Vite) | worker | untracked v0 debris | phase-0 landing in worker's lane |

--- lanes I verified green this session (all raw exit=0, disk==HEAD at commit):
- tokenizer/BPE facade family: 293 passed
- tts facade: 50 passed
- voice router facade: 5 passed
