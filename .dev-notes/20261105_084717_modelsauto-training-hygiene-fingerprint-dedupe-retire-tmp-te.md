---
id: 20261105_084717_modelsauto-training-hygiene-fingerprint-dedupe-retire-tmp-te
status: done
board: 20261105_084717_modelsauto-training-hygiene-fingerprint-dedupe-retire-tmp-te
title: "models/auto-training hygiene: fingerprint-dedupe + retire tmp_/test scratch artifacts (NEEDS USER OK before any deletion)"
tags: []
sprint: ""
gh: null
---

# models/auto-training hygiene: fingerprint-dedupe + retire tmp_/test scratch artifacts (NEEDS USER OK before any deletion)



SCOPE REALITY (read-only inventory 2026-11-05, before any deletion): fingerprint scan of models/ via artifact_registry = 32 artifacts / 4.02GB, of which hf-cache is 3.51GB (2 files, external mirror — OUT OF SCOPE) and unique models carry ~0.5GB. Duplicate-fingerprint groups = 1 (x11 tiny 0.0MB test souls) — dedupe reclaims ~ZERO bytes; this card is NAMESPACE HYGIENE, not GB recovery. Scratch-named files = 7 (tmp_*.soul ~0.2MB + dated/nometa/list_test/test/test_ckpt/dl_test souls ~0MB) plus similar .points.json; models/ is NOT git-tracked (git ls-files models/ = 0). Proposed action pending user OK: delete the 7 scratch files + stale .meta.json/.prov.json orphans, optionally relocate real trained models under models/trained/; no bulk deletes, preserve user-created models.
DONE 2026-11-05 — user gave explicit OK (card-question). models/ is not git-tracked, so cleanup = MOVE to backup, fully reversible, no hard delete. 90 files (~335 MB) moved to data/backups/models-cleanup-20261105/models/: 22 stale comprehensive/tmp_* point-snapshot pairs (Sept), auto-training/turbo junk stubs (test/evil/dated/nometa/etc), byte-identical dup stubs, genuine orphan sidecars. DEFECT FOUND+FIXED: orphan-sidecar classifier stripped .soul.meta.json too eagerly and swept 16 sidecars of REAL models (incl. today's slonet-native/qt_* x3) — restored to models/ same session; verify: backup sidecars-with-live-base = EMPTY. models/ now 48 non-hf files (was 65ish), remaining scratch-pattern files: NONE. Backup kept until user confirms.
