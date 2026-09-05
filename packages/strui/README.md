# strui (`@sloughgpt/strui`)

Standalone UI package: **SloughGPT web design** (pastel lattice tokens, `sl-*` utilities, Radix + CVA). **Primitives** match `apps/web/components/ui`. **Composed** layouts and patterns live under `src/components/composed/` (also exported as `@sloughgpt/strui/composed`). **AI** flows (`src/components/ai/`) include chat shell, model picker, RAG citations, reasoning panel, sources, attachments, and more.

Keep `src/styles/globals.css` aligned with `apps/web/app/globals.css` when the shell changes.

## Responsive Design

Build **mobile-first**: default styles target narrow viewports, then layer larger breakpoints.

### Breakpoints (Tailwind defaults)

| Breakpoint | Width | Target |
|-----------|-------|--------|
| (none)    | <640px | phones |
| `sm:`     | ≥640px | phablets, large phones |
| `md:`     | ≥768px | tablets portrait |
| `lg:`     | ≥1024px| tablets landscape, small laptops |
| `xl:`     | ≥1280px| laptops, desktops |
| `2xl:`    | ≥1536px| large desktops |

### Common patterns

```tsx
// Sidebar: hidden on mobile, shown on desktop
<div className="hidden lg:block">...</div>

// Grid: 1 col mobile → 2 col tablet → 4 col desktop
<div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4">...</div>

// Stack on mobile, row on desktop
<div className="flex flex-col sm:flex-row">...</div>

// Padding: tight mobile → comfortable desktop
<p className="px-3 sm:px-4 md:px-6">...</p>
```

### Mobile utilities (PWA / installed web)

- **`str-safe-top`** — `padding-top: env(safe-area-inset-top)` — fixed header
- **`str-safe-bottom`** — `padding-bottom: env(safe-area-inset-bottom)` — bottom composer
- **`str-safe-x`** — horizontal safe areas — full-bleed sections
- **`str-safe-all`** — all four edges — modal sheets
- **`str-min-h-screen`** — `min-height: 100dvh` — stable mobile viewport
- **`str-touch-target`** — `min-height/min-width: 44px` — Apple HIG / Material touch target
- **`str-chat-scroll`** — momentum scrolling, no scroll chaining

### Viewport meta

In the host app HTML:

```html
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
```

This enables `env(safe-area-inset-*)` on notched iOS and Android devices.

### Typography scaling

Use smaller text on mobile, larger on desktop:

```tsx
// Headings
<h1 className="text-lg sm:text-xl md:text-2xl">...</h1>

// Body
<p className="text-sm sm:text-base">...</p>

// Captions
<span className="text-xs sm:text-sm">...</span>
```

### Width constraints

Constrain wide content on mobile:

```tsx
// Selects, inputs
<input className="max-w-[70px] sm:max-w-[90px]" />

// Message bubbles
<div className="max-w-[85%] sm:max-w-[75%]">...</div>
```

### Composed (apps & dashboards)

`PageHeader`, `AppShell`, `NavRail` / `NavRailLink`, `Toolbar`, `KpiGrid`, `StatCard`, `FormField`, `SettingsRow`, `SearchInput`, `InlineBanner`, `Skeleton`, `EmptyCard`, `CopyButton`, `KeyValueList`, `ProgressBar`, `StepIndicator`, `Chip`, `SectionHeader`, `ScrollPanel`, `ListRow`, `StatusDot`, `FoldSection`, `Timeline`, `Breadcrumbs`, `Kbd`.

### AI (agents & assistants)

`ChatLayout`, `ModelPicker`, `Citation`, `ReasoningPanel`, `SourceList`, `AttachmentChip`, `StreamingAssistantPlaceholder`, `JobStatus`, plus `ChatThread`, `PromptComposer`, `MessageBubble`, `ToolCallCard`, `TokenMeter`, etc.

Storybook is the design reference for strui:

- **Docs → Introduction** — brand overview and how to use the book.
- **Docs → Design principles** — accessibility, spacing, and Storybook workflow notes.
- **Docs → Foundations** — color roles and typography (Outfit + JetBrains Mono, `sl-h1` / `sl-h2`).
- **Docs → Design principles** — accessibility, spacing, responsive, motion, and content conventions.
- **Docs → Component gallery** — one scrollable page of primitives, composed blocks, and AI surfaces; use the **Surface** toolbar (sun/moon) for light/dark.
- Per-component stories under **UI**, **Composed**, and **AI** include **Controls** (props) and **Docs** (API tables).

The manager UI uses the same lavender accent as the lattice; the canvas uses a soft mesh gradient behind stories.

## Commands

```sh
cd packages/strui && npm ci && npm run storybook
```

```sh
npm run test
```

```sh
npm run chromatic
```

## Chromatic

The strui Storybook is published to **Chromatic** from CI for visual review on pull requests (and baseline updates on `main`). The published build serves as the hosted Storybook you can share.

- CI reads the **`CHROMATIC_PROJECT_TOKEN`** GitHub Actions secret — add the token from your Chromatic project to the repo secrets.
- Locally: `npm run chromatic` publishes the current working tree (token via `CHROMATIC_PROJECT_TOKEN` env var).
- The `test-strui` CI job now also runs `npm run build-storybook`, so a broken Storybook fails CI before publishing.

## Consume

```tsx
import { Button, ChatThread, PromptComposer, cn } from '@sloughgpt/strui'
import '@sloughgpt/strui/styles/globals.css'
```

Subpath barrels: `@sloughgpt/strui/ai`, `@sloughgpt/strui/composed` (subset of the root export; root re-exports everything).

### `apps/web` (this repo)

The Next app depends on **`file:../../packages/strui`**, uses **`transpilePackages: ['@sloughgpt/strui']`**, and keeps stable imports via thin client shims under **`apps/web/components/ui/*`** that re-export from `@sloughgpt/strui`. **`lib/cn`** re-exports `cn` from the same package. **Do not** import `@sloughgpt/strui/styles/globals.css` in the web app: the shell keeps **`app/globals.css`** with its own token format (RGB triplets for `/opacity`); Tailwind **`content`** includes `../../packages/strui/src/**` so classes from the library are not purged.

## PWA and mobile web

This is **one React + Tailwind codebase** for browsers (including installed PWAs). It is not React Native; use the same components in Capacitor/Tauri/WebView shells.

- In the host app, use **`viewport-fit=cover`** (and optional `theme-color`) so `env(safe-area-inset-*)` applies.
- Use **`str-safe-top` / `str-safe-bottom` / `str-safe-x` / `str-safe-all`** on fixed headers, bottom composers, and full-bleed layouts.
- **`str-touch-target`** enforces a 44×44 minimum (send buttons, primary actions).
- **`str-min-h-screen`** uses `100dvh` for stable mobile viewport height.
- **`str-chat-scroll`** enables momentum scrolling and avoids scroll chaining on chat regions.

Storybook: **AI → ChatShell → iPhone** for viewport preview; **Composed → Overview** for app shell patterns.
