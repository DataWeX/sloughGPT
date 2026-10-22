# Intake Sizing — Card `96d1507c` "Chart quality / shared chart layer"

> Produced by product-designer intake, 2026-10-22. Card: `96d1507c-d754-44d6-b03d-8be4d2bfdd46`.
> Build order: **a → b1 → a2 → c → d**; **b2 (monitoring hero) is blocked** until a
> monitoring journey heading exists in `docs/UX_FLOWS.md` + `domain/journeys/flows.py`.

**Summary:** The chart estate is 11 recharts-importing files (~16 individual charts) plus 4 hand-rolled charts, all living on grandfathered pages (`apps/web/lib/page-baseline.json` → `"journeys": []`, gate: `apps/web/lib/page-journey-gate.test.ts`), so the Page Rule is satisfied for free as long as this card adds **zero** `page.tsx` files. Journey coverage is lopsided: the **training** half of the card has a hard citation (`12-training` → `/training`, spec `### Training: 3-Click Flow`, which explicitly names the live loss chart), while the **monitoring** half has **no journey anywhere in `docs/UX_FLOWS.md`** — that is the blocking intake gate, and slice (b)-monitoring cannot enter a lane until a monitoring flow heading is added to `docs/UX_FLOWS.md` + `domain/journeys/flows.py`. There is **no shared chart layer and no chart dir today** (verified: `apps/web/components/charts/` does not exist; `@sloughgpt/strui` is deliberately zero-dependency and has no recharts), so the foundation belongs app-local in `apps/web/components/charts/`. Honest sizing: **A = M, A2 = L, B-training = M, B-monitoring = L (blocked), C = S, D = M** ⇒ card total **L**; recommend splitting into two cards (training-chart-quality unblocked now; monitoring gated). Screenshot canary today is capture-only (Cypress `cy.screenshot`, no pixel diff, no Playwright anywhere) — the cheapest honest canary is the already-wired Cypress CI job plus a pixel-diff plugin, with vitest DOM/token assertions as the zero-flake floor.

---

## 1. Journey citations

### ✅ Training — cited and unblocked (two headings, one executable flow)

1. **`### Training: 3-Click Flow`** (`docs/UX_FLOWS.md:407-472`) — step 3 "**See results**": *"Live loss chart while training runs (visual: line going down = good)"*; and the translation table: *"Loss → 'Learning progress' (visual chart, not numbers)"* (`docs/UX_FLOWS.md:470`).
   - **Executable flow:** id **`12-training`**, label **"Train My AI (3-click)"**, url **`/training`**, spec **`### Training: 3-Click Flow`** — `domain/journeys/flows.py:115-121`; integrity enforced by `tests/test_ux_flows_library.py`.
   - **Hard design constraints from this flow for the hero view:** *"What they should NOT see: … 'training analytics'"* and *"No separate sub-pages for trends, compare, model-card, or grid-search"* (`docs/UX_FLOWS.md:346,458`) ⇒ the analytical/hero loss view **must compose into `/training` step 3**, never a new route.
   - Progressive-disclosure hook for brush/zoom: power-user extra *"View detailed loss chart with step numbers"* behind the **Advanced** toggle (`docs/UX_FLOWS.md:352`).
2. **`### 12. Train My AI`** (`docs/UX_FLOWS.md:319-354`) — user flow step 4: *"Watch the loss chart go down (visual proof it's learning)"*.

### ❌ Monitoring — NO journey exists (this is the blocking gate)

- `docs/UX_FLOWS.md` has **no monitoring / system-health / operator flow**: features 1–11 are chat-mode flows, feature 12 is training; the only other headings are `## Navigation & Layout (Plain English)` (`:357`) and `## Design Principles` (`:387`) and `### Training: 3-Click Flow`. Verified executable registry too: `domain/journeys/flows.py:124-208` `FLOWS` = `0-home, 1-chat … 11-wellness, 12-training` — **no monitoring entry**.
- The only "citation" available today is a nav route, not a journey: `/monitoring`, label `nav.monitoring`, description *"System status & errors"* (`apps/web/lib/navigation.ts:115-121`). `docs/USER_PERSONA.md` supports an operator persona (`:74` "platform/infra engineer", `:119` shared-hardware infra as Core) but a persona is not a `UX_FLOWS` journey.
- **Gate:** before any lane builds the monitoring hero (slice b2), intake must add a named flow heading to `docs/UX_FLOWS.md` (e.g. "### 13. Watch My System"), register it in `domain/journeys/flows.py` (keeps `tests/test_ux_flows_library.py` green), and cite it. Until then **b2 stays out of every lane.**

### ⚠️ No journey for the rest either
`/phoneme`, `/multimodal`, `/evaluate`, `/personality`, `/consciousness/*`, `/knowledge`, `/benchmark`, `/developer` have **no UX_FLOWS heading**. They may receive *baseline quality* (states, tokens, tooltips — slice A2) but **no hero/analytical investment** in this card, exactly as the intake note requires ("a named UX_FLOWS journey" per surface).

---

## 2. Surface inventory (verified paths)

| # | Chart component (verified file) | Renders on (verified call site) | Route | Journey mapping |
|---|---|---|---|---|
| 1 | `apps/web/components/training/LossChart.tsx` | `components/training/TrainingPipeline.tsx:19-20,235` → `app/(app)/training/page.tsx:18,224`; `components/training/JobLossCard.tsx:8-9,62` → `app/(app)/training/job/[id]/page.tsx:19,152` | `/training`, `/training/job/[id]` | ✅ `12-training` / `### Training: 3-Click Flow` |
| 2 | `apps/web/components/multimodal/TrainingCard.tsx` | `app/(app)/multimodal/page.tsx:27,226` | `/multimodal` | ❌ none (baseline only) |
| 3 | `apps/web/components/monitoring/TrendChart.tsx` | `app/(app)/monitoring/page.tsx:74-75,556` | `/monitoring` | ❌ none — **hero blocked** |
| 4 | `apps/web/components/monitoring/SystemChart.tsx` | `app/(app)/monitoring/page.tsx:66-67,676` | `/monitoring` | ❌ none — **hero blocked** |
| 5 | `apps/web/components/phoneme/HistoryCard.tsx` | `app/(app)/phoneme/page.tsx:19,326` | `/phoneme` | ❌ none |
| 6 | `apps/web/components/phoneme/ProgressSummary.tsx` | `app/(app)/phoneme/page.tsx:20,278` | `/phoneme` | ❌ none |
| 7 | `apps/web/components/compare/VisualComparisonCard.tsx` | `components/compare/ComparisonView.tsx:14` (dynamic) → `app/(app)/evaluate/page.tsx:464` | `/evaluate` (compare tab) | ❌ none |
| 8 | `apps/web/components/souls/TraitRadarChart.tsx` | `components/souls/SoulVisualizer.tsx:5,68` → `components/models/PersonalityProfileCard.tsx:8,110` → **no page imports `PersonalityProfileCard`** (grep of `apps/web/app` = 0 matches) | *(orphan chain)* | ⚠️ **dead code — flag for cleanup card, out of scope here** |
| 9 | `apps/web/app/(app)/personality/page.tsx` (inline, 2× `LineChart` at `:717,782`) | the page itself | `/personality` | ❌ none |
| 10 | `apps/web/app/(app)/consciousness/master/page.tsx` (inline, Radar `:394`, Bar `:426`, Bar `:473`, Line `:503`) | the page itself | `/consciousness/master` | ❌ none |
| 11 | `apps/web/app/(app)/consciousness/dashboard/page.tsx` (inline, Line `:501`, Area `:551`, Area `:601`) | the page itself | `/consciousness/dashboard` | ❌ none |

**Adjacent, non-recharts (hand-rolled) — explicitly out of scope, backlog only:** `components/benchmark/BenchmarkChartCard.tsx` → `/benchmark` (`page.tsx:27,220`); `components/knowledge/KnowledgeCategoryChart.tsx` → `/knowledge` (`page.tsx:70,780`); `components/startup/StartupHistoryChart.tsx` → `/developer` (`page.tsx:23,958`); inline `BarChart` → `/consciousness/statistics/page.tsx:133`.

**Page Rule verification:** every route above is in the `grandfathered` array of `apps/web/lib/page-baseline.json` (monitoring `:68`, multimodal `:69`, personality `:73`, phoneme `:74`, evaluate `:49`, training `:92`, `training/job/[id]` `:91`, consciousness dashboard `:24` / master `:32`) and `"journeys": []` (`:115`). **Zero new `page.tsx`, zero baseline edits ⇒ `apps/web/lib/page-journey-gate.test.ts` stays green and the route budget is untouched.** Do not touch the consciousness pages beyond in-place chart quality — Page Rule §4 makes them cleanup-card business, not this card's.

**Chat-lane check:** `grep 'Chart|chart' apps/web/app/(app)/chat` → **no matches**; all 11 recharts imports are outside `features/chat` and `components/chat`. **No chart sits on a chat surface, so nothing to flag as chat-out-of-scope** (keep it that way: no chart may be added to `/chat` in this card).

---

## 3. Feature boundary — shared chart layer

**Where it lives:** new `apps/web/components/charts/` — verified non-existent (`glob apps/web/components/charts/**` → none) and verified there is no existing shared chart layer (all charts are domain-local or inline). **Not** `packages/strui`: strui is zero-dependency by design (`packages/strui/package.json` deps = `class-variance-authority`, `clsx`, `tailwind-merge`, `zustand`; no `recharts`) and is consumed outside this app surface. Chart chrome (empty/error/skeleton) *already exists* in strui and must be composed, not rebuilt: `EmptyCard`, `EmptyState`, `ErrorPanel`, `Skeleton`, `Spinner`, `Tooltip` (`packages/strui/src/index.ts:148,164,165,170,175,263`).

**Proposed API (one wrapper + composition, mirroring `PageContainer`'s proven state contract):**

- `apps/web/components/charts/ChartFrame.tsx` — the only new primitive:
  - `state: 'loading' | 'empty' | 'error' | 'ready'` (or derive from `loading?: boolean`, `error?: string | null`, `onRetry?: () => void`, `empty?: boolean`) — emits `data-state` + `aria-busy` / `role="alert"` exactly like `apps/web/components/PageContainer.tsx:102,135,155,177`.
  - `label: string` (required, a11y — precedent `app/(app)/monitoring/page.tsx:671-675` `role="img" aria-label=…`), `title?: ReactNode`, `action?: ReactNode` (e.g. Export), `height?: number | 'h-40' | 'h-48' | 'h-72'`, `className?`, `children` (the recharts node).
  - `motion?: 'none' | 'enter'` (default off until slice C).
- `apps/web/components/charts/chart-tokens.ts` — `SERIES: ['var(--chart-1)', … 'var(--chart-5)']` + `AXIS = 'var(--muted-foreground)'`, `GRID = 'var(--border)'`. Pattern precedent already in-repo: `app/(app)/consciousness/dashboard/page.tsx:78-93`.
- `apps/web/components/charts/ChartTooltip.tsx` — token-styled tooltip content (`bg-popover border border-border text-xs`), replacing stock recharts tooltips.
- `apps/web/components/charts/useChartRange.ts` — optional interaction hook: emits `onRangeChange?: (range: [number, number] | null) => void` for brush/zoom. **The wrapper never slices data — the caller (feature/page) owns the derived dataset.**
- Focusable data points: keyboard-reachable point list / `tabIndex` on points inside `ChartFrame`, plus `aria-label` per point — wrapper-level, not per chart.

**MUST NOT own:** data fetching or polling (page hooks like the monitoring `liveHealth`/`fetchAll` state stay in `app/(app)/monitoring/page.tsx`); journey/step logic (`TrainingPipeline` owns the 3-step pipeline); page-level loading/error/empty (`PageContainer` owns that — chart states are *inside* a ready page); routing/layout; any new `page.tsx`.

---

## 4. Build slices (ordered, independently landable)

| Slice | Size | Contents | Charts touched |
|---|---|---|---|
| **(a) Foundation** | **M** | Create `apps/web/components/charts/` (`ChartFrame`, `chart-tokens`, `ChartTooltip`) with vitest component tests (state rendering, `data-state`, aria, no-hex token assertion). Prove it on the journey-cited surface: migrate `components/training/LossChart.tsx` (loading/empty/error + tokenized series/tooltip + focusable points). | `components/training/LossChart.tsx` (+ new `components/charts/*`) |
| **(b1) Training hero analytical view** | **M** | Range-select/brush (or zoom) over the loss series, composed into **existing** `/training` step-3 "See results" and `/training/job/[id]` — advanced detail behind the existing Advanced/progressive-disclosure pattern (`docs/UX_FLOWS.md:352`). No new route. | `LossChart.tsx`, `components/training/TrainingPipeline.tsx`, `components/training/JobLossCard.tsx`, `app/(app)/training/page.tsx` (compose only) |
| **(b2) Monitoring hero analytical view** | **L** — **BLOCKED** | Brush/zoom on the health/memory trend + real-time chart. **Cannot start until a monitoring heading exists in `docs/UX_FLOWS.md` + `domain/journeys/flows.py`.** | `components/monitoring/TrendChart.tsx`, `components/monitoring/SystemChart.tsx`, `app/(app)/monitoring/page.tsx` (compose only) |
| **(a2) Quality sweep — states/tokens/tooltips on remaining charts** | **L** | Wrap the rest of the estate in `ChartFrame`: token colors, tooltips, loading/empty/error (incl. removing silent `{loaded && …}` blanks — `app/(app)/monitoring/page.tsx:549,656`). No interaction, no motion. | `components/multimodal/TrainingCard.tsx`, `components/phoneme/HistoryCard.tsx`, `components/phoneme/ProgressSummary.tsx`, `components/compare/VisualComparisonCard.tsx`, `components/monitoring/TrendChart.tsx`, `components/monitoring/SystemChart.tsx`, `app/(app)/personality/page.tsx` (2 charts), `app/(app)/consciousness/master/page.tsx` (4), `app/(app)/consciousness/dashboard/page.tsx` (3) — **total 9 files / 14 charts** |
| **(c) Motion** | **S** | Chart enter/update motion via `ChartFrame motion` + recharts `isAnimationActive` gated by a `prefers-reduced-motion` hook and `motion-reduce:` variants; durations per design system (§5). Never gratuitous. | All charts already migrated in (a)/(a2) — no new surfaces |
| **(d) Tests + screenshot canary** | **M** | Journey-level assertions: extend `apps/web/cypress/e2e/training-page.cy.ts` (currently has no loss-chart assertion — only header/dataset-selector/history) to assert the chart + its states with `cy.mockAll()` fixtures; add the pixel-diff canary (§6). | `LossChart.tsx` + `components/charts/*` only |

Order: **a → b1 → a2 → c → d**, with **b2 gated on the journey citation** and slotted whenever that lands. **Excluded from all slices:** the orphan `TraitRadarChart`/`SoulVisualizer`/`PersonalityProfileCard` chain (no live page — send to a cleanup card), the 4 non-recharts charts, and any chat surface. Aggregate: **L**.

---

## 5. Design-system constraints (Noir Violet — `docs/design/DESIGN_SYSTEM.md`, LOCKED)

**Colors (series):**
- `--chart-1` … `--chart-5` — defined `apps/web/app/globals.css:68-72` (dark block `:129-133`, aura variants `:1070-1074,1099-1103`) and canonically `packages/strui/src/styles/globals.css:92-96,126-130,416-420,444-448`; values table `docs/design/DESIGN_SYSTEM.md:85-89`.
- **Chart rule (verbatim, `DESIGN_SYSTEM.md:91`):** every series/swatch references a token — `stroke="var(--chart-1)"`; *"Color constants at the top of a component (`const COLORS = { a: '#6366f1' }`)"* are **forbidden**. In-repo precedent: `app/(app)/consciousness/dashboard/page.tsx:78-93` (`competence: 'rgb(var(--chart-1))'` …). Forbidden-pattern row: `DESIGN_SYSTEM.md:354`.
- Semantic tokens for state/annotation: `--success`, `--warning`, `--destructive`, `--info`, `--primary`, `--muted-foreground`, `--border`, `--card`, `--popover`, `--ring` (tables `DESIGN_SYSTEM.md:39-79`; Tailwind bindings `apps/web/tailwind.config.js:31-66` incl. `chart.1..5` → `bg-chart-1`, `:67-73`).
- Ambient/theme: `--primary`, `--border` shift per aura (`DESIGN_SYSTEM.md:95-107`) — hence **no hue may be pinned in markup** (`DESIGN_SYSTEM.md:15`).

**Spacing / radius / type:**
- Card padding `p-4`/`p-6`, header→content `space-y-2`, gaps `gap-2`/`gap-4`; **no arbitrary values** (`DESIGN_SYSTEM.md:149-163`). Radius `--radius` = 6px, `rounded`/`rounded-lg` (`:166-173`; `tailwind.config.js:95-102`).
- Type: caption `text-xs text-muted-foreground`, body `text-sm`, chart-card label style `text-xs font-medium uppercase tracking-wider` (already used at `app/(app)/monitoring/page.tsx:551`); **never `text-lg` in body, never `text-2xl` outside `AppRouteHeaderLead`** (`DESIGN_SYSTEM.md:119-135`). Numeric axis/tick font via `font-numeric` (`tailwind.config.js:26-28`).

**Motion:**
- `transition-smooth` = `cubic-bezier(0.4, 0, 0.2, 1)` (`tailwind.config.js:22-24`); `duration-150` micro-interactions, `duration-200` page transitions; **"No bouncing, no spinning (except loading spinners), no flashy effects"** (`DESIGN_SYSTEM.md:330-335`).
- **`prefers-reduced-motion` precedents to follow (pick one, don't invent):**
  1. Tailwind `motion-reduce:animate-none` — 24 in-repo usages, e.g. `apps/web/app/(app)/page.tsx:272`, `apps/web/features/chat/components/TokenBilling.tsx:112`; mandated by `DESIGN_SYSTEM.md:352`.
  2. CSS media blocks: `apps/web/app/globals.css:556-564` (kills status-bar transitions) and `:857-879` (kills `.sl-bar-shimmer`, `.sl-dot-pulse`, `.sl-toast`, `.sl-interactive`); `packages/strui/src/styles/globals.css:583`.
  3. `tailwindcss-animate` (`tailwind.config.js:111`) + the documented `animate-in fade-in (respects prefers-reduced-motion)` page pattern (`apps/web/components/PageContainer.tsx:68`).
  → For recharts (which animates internally), slice (c) must additionally gate `isAnimationActive`/`animationDuration` on a `matchMedia('(prefers-reduced-motion: reduce)')` hook — the CSS precedents alone cannot disable recharts' SVG transitions.

**Enforcement hooks already documented:** `DESIGN_SYSTEM.md:402-414` (grep for hex in chart series: `grep -rn "': '#" apps/web/components apps/web/app --include='*.tsx'` must be zero).

---

## 6. Screenshot canary feasibility

**What exists (verified):**
- **Cypress 14** e2e, not Playwright: `apps/web/cypress.config.ts` (`baseUrl http://localhost:5173`, `screenshotOnRunFailure: true`); scripts `e2e`, `e2e:vite`, `ci:e2e` (`apps/web/package.json:37-43`); **already in CI**: `.github/workflows/ci.yml:66-94` job `frontend-e2e` → `npx cypress install` + `npm run e2e:vite`. `glob **/playwright*` → **none**; no `@playwright/test` in any `package.json`.
- Screenshot commands: `apps/web/cypress/support/visual-commands.ts` (`screenshotPage`, `screenshotInteraction`, `screenshotElement`, `screenshotSequence`) — **capture only**.
- Existing visual specs: `apps/web/cypress/e2e/visual-headers.cy.ts` (56 routes incl. `/training`, `/monitoring`), `visual-interactions.cy.ts`, `layout-overflow.cy.ts`, plus journey-adjacent `training-page.cy.ts`, `monitoring-page.cy.ts`. Determinism is available: `cy.mockAll()` from `apps/web/cypress/support/api-mocks.ts` (used in `visual-headers.cy.ts:60`).
- **Gap — no diffing exists:** `grep 'toHaveScreenshot|image-snapshot|pixelmatch|toMatchSnapshot'` over `apps/web` → **0 matches**; no committed baselines (`apps/web/cypress/**/snapshot*/**` → none). Today's "visual tests" would pass even on a broken chart as long as the DOM node exists.

**Recommendation (cheapest honest canary, in order):**
1. **Floor (free, do in slice a):** vitest DOM/token canaries in `components/charts/*.test.tsx` — assert `data-state`, `aria-*`, series colors resolve to `var(--chart-N)`, and zero hex (`docs/design/DESIGN_SYSTEM.md:405,408` greps as a test). No flake, runs in `npm run test` / `ci:quick`.
2. **Pixel canary (the real "screenshot canary"):** stay on **Cypress**, add **`cypress-image-snapshot`** (jest-image-snapshot + pixelmatch) as an `apps/web` devDependency, wire it in `cypress.config.ts` `setupNodeEvents`, and add one spec (e.g. `apps/web/cypress/e2e/visual-charts.cy.ts`) that visits `/training` and `/monitoring` with `cy.mockAll()` fixtures and `toMatchImageSnapshot` on the chart element with a small `maxDiffPixels` threshold. Committed baselines under `apps/web/cypress/snapshots/`, regenerated only via an explicit `ACCEPT_SNAPSHOTS=1` run, running in the existing `frontend-e2e` CI job (no new workflow).
   - *Why not Playwright `toHaveScreenshot`:* it is the better tool, but there is **zero Playwright infra** — config, install, a second browser binary and a second CI job — i.e. strictly more surface than one Cypress plugin, and the environment's browser tooling is flaky today.
   - *Known honesty limits to state on the card:* recharts renders crisp SVG (good — no canvas AA noise), but fonts (`--font-rubik`/`--font-lato`) and theme must be pinned: run snapshots only in CI's fixed OS/viewport, with light theme forced; animated charts must render with motion disabled (`reducedMotion`) or the diff will thrash — another reason slice (c)'s reduced-motion hook is a prerequisite for slice (d).
