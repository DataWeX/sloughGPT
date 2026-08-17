# Mobile App Plan — SloughGPT (`apps/mobile`)

Status: PLAN (not started). Every checkbox below is a unit of work with its test requirement.
Target: a fully featured, production-quality mobile client that matches the web app's feature
set and ships with a design that feels native and considered.

---

## 0. Current State (audit, 2026-08-16)

| Area | State |
|------|-------|
| Engine | Bare React Native 0.86 (no Expo) — react-navigation bottom-tabs + native-stack |
| Tabs | Chat / Models / Settings — icons are **emoji** (`💬🧠⚙️`) in `<Text>` |
| Colors | Hardcoded hex in `App.tsx`, `ChatScreen.tsx` (BG_PRESETS), StatusBar, navigation themes |
| Theme | `tamagui.config.ts` light/dark + 5 accent themes, not aligned to web "Noir Violet" token structure |
| Chat | `ChatScreen.tsx` = **1440-line monolith**; features already present: streaming, regenerate, voice, image, bookmarks, stars, pins, labels, reply, forward, search sessions, offline queue |
| Services | 22 files (api-client, sse-client, voice, image/file upload, llama-rn, onnx, training, push, offline-cache, haptics, sounds, toast, drafts, quick-prompts, reactions, stars, pins, labels, bookmarks, clipboard, training-collector) |
| Stores | chat, model, settings, hybrid-inference, training-data, training |
| Screens | Chat, Models, Settings (tabs); Onboarding + Health, About, Bookmarks, Help, Training, Knowledge, Search (stacked) |
| Tests | ~50 test files exist; **cannot run — no `node_modules`** (jest/react-native/tsc absent) |
| Native | **No `android/` or `ios/` dirs, no `app.json`** — cannot build or run |

Blockers: `npm install` in `apps/mobile` (~500-1000 MB — needs bandwidth approval) and
native scaffolding generation before any build/test/E2E is possible.

---

## 1. Design Direction — "Noir Violet, native"

First gap stated by the user: *the app doesn't look as good as it should.* The mobile design's
canonical source of truth is the **desktop Noir Violet design system** (documented in AGENTS.md
and `apps/web/app/globals.css`). Reference apps are studied for **interaction patterns only** —
never their visual identity (colors, fonts, geometry are ours).

### 1.1 Canonical source: desktop Noir Violet parity

The web design system is the contract. Every token the web defines must exist identically in
`tamagui.config.ts`. Web token → Tamagui token table:

| Web role (light / dark) | Web value | Tamagui token |
|-------------------------|-----------|---------------|
| Primary | `124 82 196` / `192 170 244` | `brand` |
| Accent (terracotta) | `236 145 95` / `240 176 130` | `accent` |
| Success | `52 176 125` / `72 192 140` | `success` |
| Warning | `236 168 60` / `240 192 80` | `warning` |
| Destructive | `220 80 90` / `235 100 110` | `danger` |
| Background | `248 246 252` (cream) / `17 15 24` (charcoal) | `bg.background` |
| Card | `255 255 255` / `28 25 38` | `bg.card` |
| Muted | `244 242 248` / `38 34 52` | `bg.muted` |
| Border | `228 224 242` / `52 46 72` | `border` |
| Muted FG | `130 122 150` / `150 140 172` | `fgMuted` |
| Ring | `124 82 196` / `192 170 244` | `ring` |

Current divergences to fix (audit 2026-08-16):
- Mobile background is violet-tinted `#F5F0FF`; desktop is **cream** `#F8F6FC` (dark: mobile `#0F0D18` vs desktop charcoal `#111F18` → correct value `17 15 24` = `#110F18`).
- Mobile ships **DMSans**; desktop brand font is **Outfit** (400/500/600/700) + **JetBrains Mono** (400/500) for code. Mobile must bundle Outfit + JetBrains Mono and set Tamagui `fontFamily` tokens to match the desktop type scale.
- Mobile tabs use emoji icons; desktop uses Lucide. Mobile must use `lucide-react-native` (already a dependency).
- The 5 ad-hoc accent themes (purple/pink/ocean/sunset/mint) diverge from the single desktop brand — collapse to `brand`/`accent` semantic tokens.

Type scale parity (web class → Tamagui `fontSize`/`fontWeight`):

| Role | Web | Tamagui |
|------|-----|---------|
| Page title | `text-2xl md:text-3xl` 600 | `24/30`, 600 |
| Section title | `text-base` 500 | `16`, 500 |
| Card title | `text-base` 500 | `16`, 500 |
| Body | `text-sm` 400 | `14`, 400 |
| Caption/meta | `text-xs` muted | `12`, 400, fgMuted |
| Label | `text-xs` 500 uppercase tracking-wider | `12`, 500, uppercase, letterSpacing |
| Code | `font-mono text-xs` | `12`, JetBrains Mono |
| Badge/Chip | `text-[10px]` 500 | `10`, 500 |

Spacing/radius/shadow parity: `gap-1..4` (4/8/12/16), `p-2..4` (8/12/16), `space-y-4` → Tamagui
`space`; radius `0/4/6/8/12/full`; shadow `sm/md/lg/xl` (rgba `25 22 36` light, `0 0 0` dark)
→ Tamagui `shadow*`.

Component-state parity: every interactive element implements the web's full state set —
**Button** (default/hover/active/focus/disabled/loading), **Input** (default/hover/focus/error/
disabled/placeholder), **Card** (default/interactive/active/loading/error), **Nav** (default/
hover/active/focus), **Badge/Chip** (neutral/success/warning/error/primary), **Toggle**
(off/on/hover/disabled/focus). No desktop state is dropped on mobile.

### 1.2 Inspiration policy — take the pattern, not the look

| App | We take | We do NOT copy |
|-----|---------|----------------|
| Perplexity | Single accent color discipline; 56pt chrome with hard edges; one elevated focus moment; dense source cards | Teal accent, dark-first-only, its exact geometry |
| Claude | Flat, typography-driven assistant messages; streaming caret; warm canvas | Terracotta accent, serif type, its layout |
| ChatGPT | Persistent bottom composer; suggestion chips; minimal chrome | Its sidebar/composer proportions |
| Poe | Follow-up actions surfaced as reply options | Anything else |
| Genie | Inline message action toolbar; contextual model chip | Its styling |
| Meteor Wallet (Tamagui) | Token-driven Figma→Tamagui process discipline; adaptive theming | Its visual style |
| Bunkr (Tamagui) | Screen/sheet architecture ideas | Its visual style |

Anything inspired is re-expressed in Noir Violet tokens — accent stays violet/lilac, canvas stays
cream/charcoal, type stays Outfit. A pattern that cannot be expressed in the desktop token set is
not adopted.

### 1.3 Chrome

- **Tab bar**: Lucide SVG icons instead of emoji; 56pt + safe-area; active tab = `brand` tint,
  inactive = `fgMuted`; hard edge top border, no floating shadow. Items ≥44pt touch target.
- **Headers**: 56pt, `fg` 600 title, single trailing action max; persistent "New chat" CTA
  (`brand` filled, ≥44pt).
- **Composer**: persistent bottom input (ChatGPT pattern) with attach / voice / send in one
  raised bar; auto-grow; focus ring in `ring`; every action ≥44pt.
- **Model + soul picker**: pill chip at top of chat (Claude/Genie pattern) with `success` status
  dot — reuses the desktop "rich card" identity + metadata + status layers at row scale.

### 1.4 Messages

- Assistant messages **flat, full-width** (Claude pattern) — typography carries the visual
  weight, not bubbles. User messages as soft pills (user = `bg.muted`, assistant = transparent
  on `bg.background`).
- Streaming caret (`brand`) while generating; inline action row on assistant messages:
  copy / regenerate / thumbs / bookmark / read-aloud (Genie pattern), each ≥44pt, `fgMuted` → `fg` on press.
- Suggestion chips after a complete response (ChatGPT/Poe pattern) and on empty state
  (desktop `EmptyState` chip styling).
- ReasoningPanel (exists, Grok-style) — restyle to tokens, keep behavior.
- Feedback thumbs wire to the existing `recordFeedback` → desktop's
  `FeedbackWorkflowManager` pipeline (backend unchanged).

### 1.5 Onboarding & feedback

- Single friendly centered greeting + 3 quick steps, then into chat (Claude/Genie pattern),
  expressed in Outfit + `brand` on `bg.background`.
- Light-touch "How did that answer do?" after responses — wire to existing `recordFeedback`.

### 1.6 Design don'ts (mobile mirror of desktop rules)

- No hardcoded hex/rgb in `src/` — every color via Tamagui tokens
- No emoji as UI icons — Lucide only
- No `text-lg`+ in page body — type scale table above
- No floating-card look on full-width screens — square edges, no outer radius on chrome
- No interactive element under 44pt (primary) / 24pt (minimum) touch target
- No blank screens — skeleton/spinner/empty/error states exactly per desktop rules
- No button without hover(active)/focus/disabled states where applicable on touch

---

## 2. Phased Plan

### Phase A — Foundation (design + toolchain)

- [ ] A1. `npm install` in `apps/mobile` (approved ~500-1000 MB) — restores jest/react-native/tsc
- [ ] A2. Baseline: `npm test` green on existing ~50 test files; `npm run lint` (tsc) clean
- [ ] A3. Generate native scaffolding: `react-native init`-style `android/` + `ios/` (or Expo prebuild), `app.json`
- [ ] A4. `tamagui.config.ts` token rebuild (1.1) + remove hardcoded hex in `App.tsx`, StatusBar, nav themes
- [ ] A5. Lucide tab icons + tab bar restyle (56pt + safe area)
- [ ] A6. Design-token audit: zero hex literals in `src/` components
- [ ] A7. Split `ChatScreen.tsx` (1440 lines) into feature hooks/components (chat-list, composer, drawer, toolbar) — no behavior change

Tests: A2 baseline suite; A4/A5 component tests updated to assert token-based styles; A6 grep-based guard test; A7 refactor verified by identical existing tests.

### Phase B — Chat parity with web

- [ ] B1. Streaming SSE via `sse-client` — stream state, cancel, reconnect (exists in `chat-store`; verify/fix)
- [ ] B2. Markdown rendering parity (code blocks + copy, headings, lists, quotes) — `Markdown.tsx`
- [ ] B3. Message actions row: copy / regenerate / thumbs / bookmark / read-aloud
- [ ] B4. Suggestion chips (post-response + empty state) from `quick-prompts` service
- [ ] B5. Model + soul picker chip wired to `model-store` (list/switch soul/load checkpoint)
- [ ] B6. Voice input + image upload flows (services exist) — permission, preview, failure states
- [ ] B7. Regenerate with backend context (`storeSessionContext` parity) — `chat-store.regenerate`
- [ ] B8. Offline mode: queue sends, retry on reconnect (`offline-cache`, `retryPendingSends`)
- [ ] B9. Bookmarks / stars / pins / labels / reply / forward — verify against services, surface in UI
- [ ] B10. Slash commands in composer (web has `SlashCommandMenu`) — add mobile equivalent
- [ ] B11. Search sessions modal (exists) + global search screen parity
- [ ] B12. Session management: create/rename/archive/delete, drafts per session

Tests: store unit tests (streaming/cancel/reconnect, offline queue, regenerate), component tests
(composer, bubble actions, chips, picker), integration tests (sse-client vs mocked server).

### Phase C — Server screens (parity with web pages)

- [ ] C1. Models screen: catalog, load/unload, soul switcher + checkpoint submenu, status badges
- [ ] C2. Health/Monitoring: live metrics, model status, inference stats (services + health store)
- [ ] C3. Settings: theme (light/dark/brand), chat defaults (temp/max tokens), chat background, memory
- [ ] C4. Knowledge: list/add/delete/search, batch ops, category chips (web `/knowledge` parity)
- [ ] C5. Training: start distill/fine-tune, loss chart, checkpoint catalog with load/delete, job history
- [ ] C6. Bookmarks screen, Help, About, Onboarding polish

Tests: per-screen component tests (render/empty/error/loading states), store tests, controller
(service) tests against mocked `api-client`.

### Phase D — Native & platform

- [ ] D1. Safe areas, haptics, sounds wired to all key interactions
- [ ] D2. Push notifications (service exists) — verify on-device
- [ ] D3. Clipboard copy with feedback; file/image pickers with permission denial states
- [ ] D4. On-device inference: `llama-rn-service` + `onnx-inference-service` + `hybrid-inference-store` —
  offline/local model path, hot-swap between cloud and local
- [ ] D5. Native inference engine: update `cpp/slonet.c` + `SloNet.podspec` (see 2.1) — parity with
  canonical SloNet layout, JSI/JNI bridge, Android + iOS builds
- [ ] D6. WebGPU/WebAssembly fallback alignment with web `lib/soulnet-webgpu/*`

Tests: platform-service unit tests with RN mocks (jest-setup already stubs `NativeModules`),
integration tests for hybrid store, JSI bridge smoke test on device.

#### 2.1 Native inference engine — `cpp/slonet.c` / `SloNet.podspec` update

Current state (audit 2026-08-16): 394-line C engine, iOS-only (`#include <Accelerate/Accelerate.h>`),
transformer-only weight layout — `tok_emb` → per-layer 9 params (LN, Wq, Wk, Wv, Wo, LN, W1, W2, W3)
→ final LN → `lm_head`. Constants `SLONET_MAX_LAYERS 12`, `SLONET_MAX_SEQ_LEN 128`.
`SloNet.podspec`: iOS 15+, framework Accelerate, `-O3 -ffast-math`. **No Android build path exists.**

Canonical layout sources the C engine must match:
- Backend `packages/core-py/domains/training/slonet.py` (SloNet weights, `.sou`/`.slnc` formats)
- Web `apps/web/lib/soulnet-webgpu/weights.ts` (same layout contract, `inferArch` lstm/transformer)

- [ ] D5.1 Layout audit: verify every offset/size in `slonet_load_weights` against `slonet.py`
  parameter order (token embeddings, LN, Q/K/V/O, W1/W2/W3 feed-forward, final LN, lm_head).
  `dim_ff` computation (`n_embed * 8 / 3` rounded to 64) must match the Python/WebGPU rule. Fix drift.
- [ ] D5.2 Sampling parity: `slonet_generate` temperature / top-k / top-p and greedy path must
  match `generate_numpy_stream()` semantics; add KV cache + position-indexed RoPE to match the
  Python `KVCache`/`SloRoPE` behavior; keep generation deterministic (CPU, no accelerator).
- [ ] D5.3 Architecture coverage: webgpu `weights.ts` supports **lstm + transformer**; the C engine
  is transformer-only. Decide scope — either add LSTM layout or explicitly constrain loaded
  checkpoints to transformer arch with a clear error for LSTM `.sou`.
- [ ] D5.4 Android support: add `CMakeLists.txt` + JNI wrapper so `slonet.c` builds on Android.
  Replace the hard Accelerate dependency with a portable path (`vDSP` equivalents via
  `-D` guards → plain C loops / android `cblas`/`NEON`), since Android has no Accelerate.
- [ ] D5.5 iOS module: keep Accelerate path; add a JSI native module (C++ TurboModule) exposing
  `load/forward/generate` to JS, bridging to `hybrid-inference-store` (D4). Update `SloNet.podspec`
  (`source_files`, module map, `platform :ios`), ensure it integrates under the new native
  scaffolding (A3) — verify with `pod install` in Phase A/E4.
- [ ] D5.6 C-level tests: a `test_slonet.c` harness (or ctest) covering load/unload, param-offset
  correctness, forward determinism, top-p/top-k sampling distribution, generate round-trip, and
  cross-checked logits vs Python SloNet and WebGPU engine on the same checkpoint.

Tests: D5.2/D5.3 sampling + KV + RoPE correctness; D5.6 C harness (compiled via `cc` or Android
toolchain) run as part of the mobile suite where the toolchain exists; D5.4/D5.5 device smoke tests.

### Phase E — Quality & ship

- [ ] E1. Accessibility pass: touch targets ≥44pt, contrast AA, screen-reader labels, reduced motion
- [ ] E2. Performance: FlatList virtualization, image sizing, memoization; no jank on mid devices
- [ ] E3. E2E smoke: Maestro or Detox covering onboarding → chat → send → settings
- [ ] E4. Release builds (Android APK + iOS) and install smoke test on a device
- [ ] E5. Full suite green + `tsc` clean + journal close-out

---

## 3. Test Series (mapped to phases)

| Layer | Tool | Covers | Phase |
|-------|------|--------|-------|
| Service unit | Jest + RN mocks | api-client, sse-client, voice, image, file, llama-rn, onnx, training, offline-cache, drafts, quick-prompts, reactions, stars, pins, labels, bookmarks, clipboard, toast | A, B, D |
| Store unit | Jest (zustand) | chat-store, model-store, settings-store, hybrid-inference-store, training stores | A, B, C, D |
| Component | `@testing-library/react-native` | composer, bubbles, actions, chips, pickers, tab bar, per-screen render/empty/error/loading | A, B, C |
| Refactor guard | Jest | ChatScreen split — identical outcomes, no behavior regression | A |
| Integration | Jest + mocked fetch/SSE | streaming reconnect, regenerate context, offline queue flush, hybrid cloud↔local switch | B, D |
| Token guard | grep-based test | no hex literals in `src/components`, `src/screens` | A, C |
| Navigation | Jest + react-navigation | tabs, stack params, onboarding gate, deep-link targets | B, C |
| E2E | Maestro/Detox | onboarding → chat → send → copy → settings → models | E |

Acceptance per item: **one failing test is a failing checklist item** — every checkbox above must
have its test suite green before it is marked off.

---

## 4. Environment Notes

- `npm install` in `apps/mobile` (~500-1000 MB) requires explicit bandwidth approval first.
- Native builds require Android SDK / Xcode; not verifiable in this environment — device smoke
  test is a handoff step (E4).
- `/tmp` purge kills any CDP browser/dev-server processes between sessions — restart per repo notes.
- Mobile tests previously skipped (2026-08-06 audit): toolchain absent, not a design decision.
