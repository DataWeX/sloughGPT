# SOP — AI + Mobile App UI Development

Standard operating procedure for building UI in `apps/mobile` (bare React Native 0.86 + Tamagui) for an AI-powered product. Web-researched 2026-10-31; anchored to this repo's reality (`docs/plans/mobile-app-plan.md`, Noir Violet tokens, existing design skills).

Read with: `.opencode/skills/frontend-design` (identity), `.opencode/skills/responsive-design`, `.opencode/skills/accessibility-audit`, `.opencode/skills/component-state-completeness`.

---

## 1. The one rule

**AI features are an optional layer over controls, never a replacement for them.** Every AI affordance (chat, suggestion, auto-action) ships alongside a conventional path to the same outcome. Users must be able to edit, override, or opt out of anything the model did.

Rationale (industry consensus, 2026): automation that removes the manual path erodes trust and retention; automation that _suggests_ and yields to the human compounds it.

### 1.1 Architecture ruling (user, 2026-10-31) — own server, local-first

**The backend is our own server** (`apps/api` FastAPI + our inference engines — SloughGPT/SlonNet, llama.cpp, ONNX) — not a third-party cloud AI API. **Local-first is the philosophy**: inference and data live on our server or on-device; the mobile app is a **thin client over our contract** with a local/on-device fallback path. Generic cloud-AI-app patterns (vendor SDKs, remote-only inference, per-token billing flows) must **not** be bolted on — the app talks our contract, and anything the model does must still be possible offline against the local engine.

## 2. Mobile UI principles (apply to every screen)

1. **Thumb-zone layout** — primary actions in the bottom third; nothing critical only in the top-right corner.
2. **Visibility of system status** — every async operation (streaming, upload, training, on-device inference) has a visible, labeled state. Nielsen heuristics 1 & 4 are the two that mobile apps fail most.
3. **Progressive disclosure** — one decision per screen; advanced options behind disclosure, not stacked by default.
4. **Recognition over recall** — icons always carry labels or accessible names; never a bare glyph the user must remember (the current emoji tabs `💬🧠⚙️` are the anti-pattern — replace with real icons + labels when touched).
5. **Adaptive dark mode** — theme via tokens (light/dark), never color inversion. Matches our Noir Violet token structure.
6. **Gesture + button both** — gestures are accelerators, never the only way to reach an action; every gesture needs a tappable equivalent.
7. **No long launch animation** — users open their daily app many times; land on content, not on a splash.
8. **Accessibility is a ship gate** — 2026 App Store/Play enforcement is stricter; Dynamic Type/font scaling, contrast, 44×44pt targets, screen-reader labels are requirements, not polish (see `accessibility-audit` skill).

## 3. AI-specific UX patterns

### 3.1 Streaming (chat is our core journey)

- Render tokens as they arrive; never block the UI on completion.
- Show a stable "thinking/streaming" indicator that does **not** reflow the layout between states.
- Streamed content uses the same styles as final content — no post-processing flash/rewrap.
- Allow scroll-away with a "jump to latest" affordance; do not force-pin the viewport mid-stream.
- Errors mid-stream: keep partial output, show an inline retry — never discard what the user already read.

### 3.2 On-device inference (llama-rn / ONNX in this app)

- Treat inference output as **first-class state**, not a side effect: belief/prediction lands in the store (zustand) like any other state, and screens derive from it.
- Always surface _which_ engine answered (cloud vs. on-device) and surface loading/fallback when the device engine can't take the load.
- Model downloads and cold starts are honest progress states with cancel — never an indefinite spinner.

### 3.3 Transparency and control

- Explain-why: any AI recommendation or auto-decision gets a one-line rationale on tap.
- Every auto-action has a visible undo (snackbar/toast + action button).
- Destructive or spending actions (delete, train, overwrite) stay behind explicit confirmation — AI never gets a silent path to them.

### 3.4 Voice and multimodal

- Voice input is additive: text field always remains the fallback path.
- Permission requests happen **in context** (user taps mic → then ask), never at cold launch.

## 4. Design system enforcement (Noir Violet, mobile half)

- **Colors**: only from theme tokens (`src/theme/colors.ts`, Tamagui config) — no hex literals in components.
- **Spacing**: scale-based only (4/8-based tokens), no arbitrary pixel values.
- **States**: every interactive component ships hover _(web)_, focus-visible, pressed, disabled, loading, success — see `component-state-completeness` skill.
- **Icons**: `lucide-react-native` only; no emoji as UI iconography.
- **No AI-default patterns** (from `frontend-design` skill): no card soup, no generic gradient hero, no unstyled list of identical rows.

## 5. Build & verify loop

| Step              | Command (from `apps/mobile`)                                                                       |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| Type gate         | `npm run lint` (tsc --noEmit)                                                                      |
| Unit/integration  | `npm test` (jest; services + component tests exist — keep them green)                              |
| Android debug APK | `cd android && ./gradlew assembleDebug -PreactNativeArchitectures=x86_64`                          |
| Emulator          | `emulator -avd Pixel_API_26` (x86_64 — build must match ABI)                                       |
| Install           | `adb install -r android/app/build/outputs/apk/debug/app-debug.apk`                                 |
| Disk reality      | native builds need ~5 GB free; purge `~/.gradle/caches` + `.cxx` intermediates before blaming code |

**Journey check (the actual done-condition):** walk `docs/UX_FLOWS.md` flows in the running app on the emulator — a non-technical user can complete the chat journey end-to-end (send → stream → regenerate → history) without hitting a dead end.

## 6. Testing strategy (layered — matches web stack)

1. **Unit** — pure logic: services (`src/services/__tests__/`), stores.
2. **Component** — RTL (`@testing-library/react-native`): rendering, states, a11y labels.
3. **Journey** — the flow above; each new screen cites a `UX_FLOWS.md` journey (Page Rule applies to mobile too: no engine-trophy screens).

A test that only asserts "it renders" does not justify a screen.

## 7. Decision record — Expo vs. bare RN (parked question)

Current: **bare RN 0.86** — native code present (`cpp/SloNet.podspec`, custom Gradle, C++ engine bridge). Expo is _not_ a config toggle here; adopting it means a managed-workflow migration (prebuild/EAS, losing direct Gradle edits or moving to config plugins). Decision criteria: if we need OTA updates + store-submission automation and our native surface stays small, revisit; if SloNet/onnx native modules keep growing, bare RN stays. Any move = its own kanban card, not a side effect of UI work.

## 8. Sources (2026)

- UXPin — _What Is Mobile UI_ (2026): platform HIG, AI-assisted design, accessibility as store gate.
- onething — _UI UX Design for Mobile Apps 2026_: AI as optional layer; transparency; automation ↔ control balance.
- UX Pilot — _9 Mobile App Design Trends 2026_: adaptive/contextual UI, short motion with clear triggers.
- EasyComm — _Mobile Dev Trends 2026_: framework trade-off table; DORA "AI is a mirror and a multiplier" (it compounds testing discipline, or accelerates debt).
- Medium — _AI-First Mobile App Architecture_: inference-ready data models; inference output as state, not side effect.
- UXCam — _45 Mobile App Best Practices_: Nielsen's 10 heuristics applied to mobile.
- ThirdRock — _Mobile UI/UX 2026_: thumb-zone, progressive disclosure, anti-pattern = long splash animation; 5-user tests catch 85% of usability issues.
- Software Co — _Best Practices 2026_: layered testing unit → integration → journey; beta channel validation.
- Eleken — _Onboarding 2026_: start at the aha moment; always give a way out.

---

_Maintenance: update this file when a ruling changes (e.g. the Expo question resolves); do not fork it into a second mobile doc._
