---
id: 20261102_085248_perfweb-settings-generation-polling-spams-patchget-every-3s
status: done
board: 20261102_085248_perfweb-settings-generation-polling-spams-patchget-every-3s
title: "perf(web): settings generation polling spams PATCH/GET every ~3s"
tags: ["settings", "frontend", "performance", "polling", "cleanup"]
sprint: ""
gh: null
---

# perf(web): settings generation polling spams PATCH/GET every ~3s

Observed in journal/network logs during the meta-weight investigation (2026-11-01), separate from both chat-loop bugs.

SYMPTOM: the frontend issues PATCH /settings/generation roughly every 3 seconds with an IDENTICAL payload, alongside polling the deprecated GET /config/generation. Neither the value nor anything server-side is changing -- it is a write loop, not a reaction to change.

WHY IT MATTERS: (1) pointless write traffic + audit/log noise on a state that is not mutating; (2) a periodic identical PUT/PATCH can mask a REAL settings change in the noise (and can race/overwrite a concurrent edit from another tab or the API); (3) GET /config/generation is deprecated -- it should not be in the polling set at all.

EXPECTED: poll only when the value can actually change (server-driven updates, or focus/visibility re-sync), skip the write when the payload is byte-identical to the last acknowledged value, and drop the deprecated endpoint in favour of the current settings contract.

SCOPE: frontend settings/generation only. Do NOT touch generation params semantics, the meta-weight band (fix/meta-weight-band), or VoiceInput (fix/voice-input-duplication).

Acceptance: no identical-payload PATCH in steady state (assertable in a test with a fake timer); deprecated GET /config/generation gone from the polling path; a real settings change still round-trips and is reflected in the UI.

CLOSED 2026-11-11 (OPENCODE perf-web lane) — pushed branch `fix/settings-polling-identical-write` (54f7a8c5e impl, 4067cef32 test-mock drive-by, 9d0a8b08c AGENT_SYNC). Acceptance met: identical-payload guard (all-identical updateSettings = no trackEvent/churn/localStorage/PATCH), deprecated GET /config/generation gone from the polling path (settingsController.getGeneration), a real change still round-trips to the backend. Gates: store 37/37, hook 37/37 incl. 4 new round-trip tests, retargeted files 52/52, tsc --noEmit 0, eslint 0 errors, npm run test:changed 595 passed (49 files) exit 0 — RED 6 watched first. ChatPageSections duplicate writes were DELETED, not retargeted: model.setTemperature/setMaxTokens already PATCH via the store, so retargeting would double-PATCH the same endpoint; generation-config-controller retired, 0 references remain. Drive-by 4067cef32: training/page.test.tsx selector-only toast-store mock lacked getState() (notify.ts:41) — pre-existing (file byte-identical to HEAD, @/lib/controllers mocked wholesale) and only surfaced by the changed-closure when the backend was down; mock now mirrors the real export shape (lib/toast-store.ts:65). Nothing weakened.
