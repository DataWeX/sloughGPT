---
id: 20260924_058_ui_flows_read_my_files
status: done
title: "Build Order #5 — UI flows: Read My Files journey + training/knowledge jargon alignment"
tags: [frontend, ux, journey]
---
# Card 058 — UI flows (Build Order #5)

## Scope (from PRODUCT_ENGINEERING Build Order #5 + UX_FLOWS.md)

Restructure knowledge/training pages to follow UX_FLOWS.md journeys.

### Journey 1: "Read My Files" (UX_FLOWS feature 3) — THE core gap
Knowledge page today is a memory-management console (facts/topics/stats);
the spec journey is missing entirely:
- Hero drop zone: "Drop a file here or click to upload" (PDF/Word/text)
- Status copy: "Reading your file..." (NEVER "Indexing...")
- After upload: suggested questions ("Summarize this", "What are the key
  points?", "Explain this in simple terms") → one tap into chat Q&A
- Jargon ban: no "RAG", "vector store", "chunks", "ingestion" visible

### Journey 2: "Train My AI" (feature 12) — copy-level gaps only
Page already matches spec ("Teach me", 3-step pipeline, resume affordances).
Violations found on main:
- Hero says "Turbo training live/finished/failed" (spec: no mode names)
- KPI "Corpora" (jargon — spec says datasets)

### Journey 3: phoneme — OUT OF SCOPE (documented)
Page exists but is nav-hidden (fd27316be removed nav entries; NAV_SECTIONS
has no /phoneme). Treated as infra for Talk/Read per fd27316be; not a
journey blocker, no UX_FLOWS spec references it.

## Approach (locked after recon)
1. **kb.py `ingest_file`**: add PDF text extraction (PyMuPDF/PyPDF2
   fallback, same strategy as domain/training data_import) so PDFs work;
   .docx has no server-side parser — frontend gives a friendly message.
2. **http-client.ts**: add `apiPostForm` (multipart FormData) — single
   source of truth rule; no raw fetch().
3. **kb-controller.ts**: `ingestFile(file, topic)` → POST /knowledge/ingest-file.
4. **Knowledge page**: new top `ReadMyFilesCard` hero (drop zone,
   "Reading your file...", success chip + 3 suggested-question chips →
   `/chat?q=...`). Rename fold heading "Advanced — memory, RAG & topics"
   → "Advanced — memory & topics".
5. **ChatPage.tsx**: read `?q=` once on mount → dispatch existing
   `send-text` CustomEvent (same mechanism VisionTabContent uses; child
   effects register listener before parent effect runs) → strip param.
6. **Training page**: Turbo→plain hero copy, Corpora→Datasets.

## Work location
Worktree `/home/mana/Documents/sloughgpt-ui-flows`, branch
`feat/ui-flows-read-my-files` (isolated off main; shared tree has other
sessions' WIP — 464 dirty files).

## Gates
- ruff (apps/api), tsc --noEmit + vitest (apps/web), pytest (kb router),
  targeted page tests updated for copy changes.

## Result
Merged to main as `63fa3bfbb` (2026-09-25).

- Knowledge page opens with **ReadMyFilesCard** hero: drop zone
  ("Drop a file here or click to upload"), "Reading your file..." status,
  result line + 3 spec-exact suggested questions -> `/chat?q=` deep link.
- `kb.py ingest_file`: PDF text extraction (pypdf preferred, PyMuPDF
  fallback; `pypdf>=4.0.0` added to requirements.txt), .docx/.doc
  friendly-rejected (no server-side Word parser — documented deviation
  from spec's PDF/Word/text promise).
- `http-client.ts apiPostForm` — multipart FormData, no JSON content-type.
- `ChatPage` handles `?q=` once via existing `send-text` CustomEvent,
  strips the param (window.location.search as source of truth).
- Training jargon purge: page hero + toasts + browser notifications +
  QuickTrain/Turbo card copy ("Turbo*"/"Corpora"/"Hyperparameters" gone);
  knowledge fold heading "Advanced — memory & topics".
- Gating fixes in touched files: pre-existing no-extra-semi ×2,
  prefer-const + no-constant-condition ×2 in http-client, React-19 JSX
  namespace import in ConsciousnessNotificationsPanel.

Gates: `tsc --noEmit` exit 0; eslint 0 errors; full vitest 826 files pass,
6 files / 32 tests fail = **exact clean-main baseline** (verified by
running the same 6 files on detached main 1ef601bd6); pytest
tests/server/test_kb_router.py 47/47 (post-merge with card 056 fixes);
ruff clean. No benchmark applies (UI-only change).

Out of scope (documented): phoneme page — already nav-hidden infra for
Talk/Read per fd27316be; not referenced by UX_FLOWS.md.

## Result (2026-09-26)
Verified landed on main (`63fa3bfbb` merge of `f0603520b`, ancestor of main):
`apps/web/components/knowledge/ReadMyFilesCard.tsx`, `apiPostForm` in
`apps/web/lib/http-client.ts`, `kb-controller.ts` + PDF extraction
(`pypdf`, tests in `test_data_import.py`) all present on main.
Re-ran gates on current main (fix-recovery checkout): vitest
ReadMyFilesCard + knowledge page + kb-controller = **48/48 pass**
(run from `apps/web/` — root lacks the `@` alias).
Status closed: journey shipped as scoped.
