---
id: 6cba481c
title: "25ebfc0f step 2 RE-SCOPED: '29 raw fetch() bypasses' is wrong on three counts — count is 40, all 3 frontends already have clients, and most sites are deliberate"
status: review
board_id: Main
date: 2026-10-16
branch: fix/fetch-timeout-audit
tags: [fetch-audit, timeouts, web, mobile, sdk-ts, 25ebfc0f]
---

# Card 6cba481c — the fetch audit (DELIVERED)

Branch `fix/fetch-timeout-audit` (worktree `.wt-fetchaudit`), 3 commits:

- `12a594e45` ← ported from `d54267a38` — sdk streaming endpoints get the
  request timeout they never had (was stranded on `fix/sdk-stream-timeout`
  plus cherry-picks `b3f7309e9` / `1e43fc857`; reached no trunk).
- `63872b4ae` ← ported from `77e80ad56` — mobile image-upload + onnx
  checkpoint download bounds (was stranded on `feat/domain-port` /
  `feat/generate-consolidation`; reached no trunk).
- `7026d974d` — the two genuine gaps this audit found, fixed here.

## Method

Exact card regex `(?<![\w.$])fetch\s*\(` over tracked `.ts/.tsx` in
apps/web, apps/mobile, packages/sdk-ts (worktree: no node_modules noise).
25 + 17 + 1 raw hits; 6 web hits are comments/docstrings/code-generation
strings (non-calls) → **37 real call sites: web 19, mobile 17, sdk 1** —
matches the card's measured 19/17/4→1 exactly.

## Verdicts — 35 justified in place, 2 genuine (both fixed)

### apps/web (19)

| Site | Class | Why it stays |
|---|---|---|
| consciousness/debug/page.tsx:243 | external-target | Hand-typed arbitrary-host URL; deliberately bypasses http-client so auth headers never reach a third party (in-code comment) |
| useVisionStudio.ts:158, 189 | local | `previewUrl` = FileReader `readAsDataURL` result → `data:` URL, no network |
| useChatLocalEngine.ts:30 | streaming/asset | Model `.soul` up to 500 MB; a fixed timeout would kill legitimate downloads |
| useErrorStream.ts:119 | diagnostic | Bounded `AbortSignal.timeout(5000)` |
| useV86.ts:66, 72 | streaming/asset | HEAD probe + fallback, both bounded `IMAGE_PROBE_TIMEOUT_MS` |
| dev-log.ts:148 | diagnostic | Fire-and-forget ingest |
| error-reporter.ts:86 | diagnostic | Fire-and-forget; must not depend on the client it reports |
| http-client.ts:644, 686, 1344, 1380 | client layer | IS the request layer (timeout/retry/circuit-breaker live here) |
| multimodal-controller.ts:76 | local | `data:` URL decode |
| **planner-proxy.ts:31** | **genuine → FIXED** | Server→server to `127.0.0.1:8787`, unbounded; now `AbortSignal.timeout(15s)` → existing catch → 502 |
| soulnet-webgpu engine:194, transformer:345, worker:47 | streaming/asset | Binary weight loads, unbounded size by design |
| sse-client.ts:154 | streaming | `getReader()` stream |

### apps/mobile (17)

| Site | Class | Why it stays |
|---|---|---|
| useConnectionStatus.ts:44 | diagnostic | Bounded (controller.signal) |
| useOnlineStatus.ts:19 | diagnostic | Lightweight health ping — card precedent calls these deliberate |
| OnboardingScreen.tsx:93 | diagnostic | Bounded |
| api-client.ts:127, 214 | client layer | IS the request layer (AbortController timeout) |
| image-upload.ts:135 | bounded (carried) | `77e80ad56` → `63872b4ae` |
| **llama-rn-service.ts:96** | **genuine → FIXED** | HEAD probe of our `/models/download/qwen-gguf`, unbounded; a zombie server hung the load flow before the HF fallback. AbortController + 10 s, mirroring voice-input.ts; abort lands in existing catch → HF fallback |
| onnx-inference-service.ts:464 | bounded (carried) | `77e80ad56` → `63872b4ae` |
| providers-client.ts ×6 (121, 219, 284, 385, 449, 545) | external-target | Per-vendor baseUrl/apiKey/wire protocol; routing through api-client would send OpenAI traffic to our own backend |
| sou-loader.ts:317 | local | `file:` URI fallback when FS module absent |
| sse-client.ts:33 | streaming | Stream reader |
| voice-input.ts:92 | bounded | AbortController + `TRANSCRIBE_TIMEOUT_MS` (pattern source for the llama fix) |

### packages/sdk-ts (1)

| Site | Class | Why it stays |
|---|---|---|
| client.ts:225 | client layer | `fetchResponse()` owns timeout + ok-check + AbortError→408 (post `d54267a38`) |

## Gates

web: vitest planner-proxy **5/5**, `tsc --noEmit` 0, eslint 0 ·
mobile: jest llama-rn **4/4**, image-upload+onnx **12/12**,
`tsc --noEmit` 0 (no eslint config exists in apps/mobile — none to run) ·
sdk: **231/231**, tsc 0 (matches `d54267a38`'s upstream numbers exactly).

## Findings for the queue (mined, not just pass/fail)

1. **Dead-code candidate (needs sign-off, NOT done here):**
   `apps/web/app/api/planner` route handlers + `lib/planner-proxy.ts` sit in
   the Next `app/` tree, but `next` is not a dependency of apps/web (only its
   `next-auth` peer remains) and there is no `next.config.*`; Vite is the web
   stack (card 85e15cc4). The proxy's only in-repo consumer is its own test.
   Bounded here (cheap, safe either way); deletion is a separate card.
2. **Health-ping asymmetry**: `useConnectionStatus` bounds its `/health`
   ping, `useOnlineStatus` does not. Card precedent says lightweight-ping
   deliberate → left alone; a one-line consistency hardening if wanted.
3. Both source fixes were stranded on 2-4 branches each before this branch —
   the port-don't-rebuild pattern keeps repeating for this workstream.
