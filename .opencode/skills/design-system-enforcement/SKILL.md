---
name: design-system-enforcement
description: Enforce the Noir Violet design system. Scans for hardcoded colors, arbitrary spacing, missing hover states, broken type scale, and token violations.
---

# Design System Enforcement Skill

## When to Use

- Pre-commit checks on UI files
- Reviewing PRs that touch components
- Creating new components or pages
- After bulk UI changes

## Design Tokens

### Colors (RGB triples)

| Token                | Light         | Dark          | Tailwind                             |
| -------------------- | ------------- | ------------- | ------------------------------------ |
| `--primary`          | `124 82 196`  | `192 170 244` | `text-primary`, `bg-primary`         |
| `--accent`           | `236 145 95`  | `240 176 130` | `text-accent`, `bg-accent`           |
| `--success`          | `52 176 125`  | `72 192 140`  | `text-success`, `bg-success`         |
| `--warning`          | `236 168 60`  | `240 192 80`  | `text-warning`, `bg-warning`         |
| `--destructive`      | `220 80 90`   | `235 100 110` | `text-destructive`, `bg-destructive` |
| `--background`       | `248 246 252` | `17 15 24`    | `bg-background`                      |
| `--card`             | `255 255 255` | `28 25 38`    | `bg-card`                            |
| `--border`           | `228 224 242` | `52 46 72`    | `border-border`                      |
| `--muted`            | `244 242 248` | `38 34 52`    | `bg-muted`                           |
| `--muted-foreground` | `130 122 150` | `150 140 172` | `text-muted-foreground`              |

### Chart Tokens

Every chart series, progress bar, and per-item accent uses a chart or semantic token — never hex, never stock Tailwind.

| Token       | Light        | Dark          | Use                                                |
| ----------- | ------------ | ------------- | -------------------------------------------------- |
| `--chart-1` | `124 82 196` | `192 170 244` | `stroke="var(--chart-1)"`, `fill="var(--chart-1)"` |
| `--chart-2` | `52 176 125` | `72 192 140`  | Series 2 / success family                          |
| `--chart-3` | `236 145 95` | `240 176 130` | Series 3 / accent family                           |
| `--chart-4` | `90 150 220` | `100 165 240` | Series 4 / info family                             |
| `--chart-5` | `220 80 90`  | `235 100 110` | Series 5 / destructive family                      |

**Rule:** a component's color map must be token references (`const C = { a: 'var(--chart-1)' }`), never hex (`{ a: '#6366f1' }`). The theme/palette switcher only works if markup never pins a hue.

### Typography Scale

| Role          | Class                                          | Size | Weight                       |
| ------------- | ---------------------------------------------- | ---- | ---------------------------- |
| Page title    | `text-2xl md:text-3xl`                         | 600  | Only in `AppRouteHeaderLead` |
| Section title | `text-base`                                    | 500  | Card headers                 |
| Body          | `text-sm`                                      | 400  | Primary content              |
| Caption       | `text-xs text-muted-foreground`                | 400  | Secondary info               |
| Label         | `text-xs font-medium uppercase tracking-wider` | 500  | Form labels                  |
| Badge         | `text-[10px] font-medium`                      | 500  | Status badges                |

### Spacing

| Token                   | Value               | Usage             |
| ----------------------- | ------------------- | ----------------- |
| `sl-page`               | `p-4 sm:p-6 md:p-8` | Page wrapper      |
| `max-w-4xl`             | 896px               | Max content width |
| `space-y-4`             | 16px                | Between cards     |
| `gap-1` through `gap-4` | 4-16px              | Component gaps    |

## Violation Scanner

### Hardcoded Colors

```bash
# Find hex colors in components (comments included — the real gate,
# apps/web/lib/design-criteria.test.ts, strips comments and ratchets)
grep -rn '#[0-9a-fA-F]\{3,8\}' apps/web/ --include="*.tsx" --include="*.css" | grep -v 'node_modules\|\.next\|dist-vite\|\.test\.\|stories\.'

# Find rgb()/rgba() not using var()
grep -rn 'rgb(' apps/web/ --include="*.tsx" --include="*.css" | grep -v 'var(--' | grep -v 'rgb(\$' | grep -v "sample: 'rgb(" | grep -v 'node_modules\|\.next\|dist-vite'
```

The mechanical ratchet lives in `apps/web/lib/design-criteria.test.ts`
(hex + numeric-rgb baselines, shrink-only). Run it with
`npx vitest run lib/design-criteria.test.ts` — a new violation fails CI,
so fix the file and delete its baseline row rather than adding rows.

**Fix:** Replace with token: `text-primary`, `bg-border`, `rgb(var(--primary))`

**Exempt (palette content, not chrome):** data-driven swatches rendering `rgb(\`...\`)`from a token data table — the magazine`Swatch`displays the raw triple as mono text +`aria-label`, so the literal *is* the content being documented. Literal `rgb('...')`/`rgb("...")`in style props are violations: wrap the scope in a token class (e.g.`.magazine-dark-preview`re-declares`html.dark`values) and use`rgb(var(--token))`.

### Arbitrary Spacing

```bash
# Find arbitrary padding/margin not in design system
grep -rn 'px-[0-9]\|py-[0-9]\|p-[0-9]' apps/web/ --include="*.tsx" | grep -v 'sl-page\|sl-app'
```

**Fix:** Use spacing tokens: `p-2`, `p-3`, `p-4`, `gap-2`, `gap-3`, `gap-4`

### Type Scale Violations

```bash
# Find text-lg (not in design system)
grep -rn 'text-lg' apps/web/ --include="*.tsx"

# Find text-2xl outside page headers
grep -rn 'text-2xl' apps/web/ --include="*.tsx" | grep -v 'AppRouteHeader'
```

**Fix:** Use `text-sm` for body, `text-base` for sections, `text-xs` for captions

### Missing Hover States

```bash
# Find clickable elements without hover
grep -rn 'onClick' apps/web/ --include="*.tsx" | grep -v 'hover:\|hover>'
```

**Fix:** Add `hover:bg-primary/10`, `hover:text-primary`, or `hover:-translate-y-0.5 hover:shadow-md`

### Inline Styles

```bash
# Find style attributes
grep -rn 'style={{' apps/web/ --include="*.tsx"
```

**Fix:** Use Tailwind classes instead

### Stock Tailwind Palette (AI-Slop Rainbow)

The single biggest palette leak: pages importing Tailwind's default color wheel and washing out Noir Violet. This found 367 instances across 35 pages, concentrated in `consciousness/*`.

```bash
# All of these must be zero before a UI commit
grep -rEo '(bg|text|border|ring|from|to|via|divide|decoration|accent|caret|fill|stroke|shadow|outline)-(indigo|fuchsia|cyan|sky|emerald|lime|amber|teal|rose|pink|red|yellow|green|blue|orange|purple)-(50|100|200|300|400|500|600|700|800|900|950)' \
  apps/web/ --include="*.tsx" --include="*.css" | wc -l
```

**Fix:** semantic utility → `text-success`, `bg-warning`, `text-destructive`, `text-primary`, `border-border`. Distinct-hue series → `var(--chart-N)`.

> **Do not flag the Wine Wash.** `ActiveModelBanner`'s `bg-gradient-to-br from-primary/[0.04] via-transparent to-accent/[0.03]` (and its `ActiveModelBannerSkeleton`), plus the `.sl-sidebar-surface` / `.sl-mobile-header` radial washes, are signature. They're token-tinted at 3–10% opacity. The rainbow ban targets stock-hue utilities (any `from-green-*`, full-opacity vivid gradients, `bg-gradient-to-*` with stock palette hues), never these.

### Hardcoded Chart / List Color Constants

```bash
# Hex constants at top of a component (chart series, per-item maps)
grep -rn "': '#" apps/web/ --include="*.tsx" | grep -v '.test.'
```

**Fix:** `{ a: 'var(--chart-1)' }`, `{ ok: 'var(--success)' }`, `{ bad: 'var(--destructive)' }`.

### Dashboard Card Soup

```bash
# Count plain-card sections per file — 5+ plain <Card> in a row, no KpiGrid, is a flag
grep -rEc '<Card>' apps/web/app/\(app\)/ --include="page.tsx" | sort -t: -k2 -rn | head
```

**Fix:** `KpiGrid` + `StatCard` for metrics, `SectionHeader` for grouping, `FoldSection` for secondary controls, `StatusDot` for live state, actions in `headerRight`.

## Component Checklist

Before committing any UI component, verify:

| Check                                                                | Pass |
| -------------------------------------------------------------------- | ---- |
| Colors use tokens (no hex, no rgb without var)                       |      |
| No stock Tailwind palette utils (`text-green-500`, `bg-red-400`…)    |      |
| Chart series / progress bars use `var(--chart-N)` or semantic tokens |      |
| Typography follows scale (no text-lg in body)                        |      |
| Spacing uses system (no arbitrary px values)                         |      |
| Interactive elements have hover states                               |      |
| Focus states use ring-2 + ring-offset-2                              |      |
| Disabled states use opacity-40                                       |      |
| Metrics use `KpiGrid` + `StatCard`, not raw `Card` + `text-2xl`      |      |
| Actions in `PageContainer headerRight`, not a body card              |      |
| Secondary controls use `FoldSection`                                 |      |
| Live state uses `StatusDot` (pulse)                                  |      |
| Cards use CardHeader/CardTitle/CardContent                           |      |
| Page uses `sl-page mx-auto max-w-4xl`                                |      |
| No inline styles                                                     |      |
| No console.log in production                                         |      |
| Component has TypeScript interface for props                         |      |
| Component has test file                                              |      |

## Anti-Patterns

| Anti-Pattern                                                        | Fix                                                                    |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `text-gray-500`                                                     | `text-muted-foreground`                                                |
| `bg-white`                                                          | `bg-card`                                                              |
| `border-gray-200`                                                   | `border-border`                                                        |
| `text-green-500` / `bg-red-400` / `text-indigo-600` (stock palette) | `text-success` / `bg-destructive` / `text-primary` or `var(--chart-N)` |
| `#6366f1` / `#22c55e` in chart series / progress bars               | `var(--chart-1)` / `bg-success`                                        |
| `const COLORS = { k: '#hex' }` in a component                       | `{ k: 'var(--chart-N)' }`                                              |
| `text-lg` in page body                                              | `text-sm` or `text-base`                                               |
| `px-8 py-6` on page                                                 | `sl-page` class                                                        |
| `rounded-lg` everywhere                                             | Use appropriate radius for context                                     |
| Stat cards as raw `Card` + `text-2xl`                               | `KpiGrid` + `StatCard`                                                 |
| "Actions" card in body                                              | `PageContainer headerRight`                                            |
| Five flat cards with no hierarchy                                   | `KpiGrid`/`SectionHeader` or rethink layout                            |
| Live dot via `bg-green-400` in a card                               | `StatusDot` (`tone="success" pulse showLabel`)                         |
| No hover on button                                                  | Add `hover:bg-primary/90`                                              |
| Focus without ring                                                  | Add `focus-visible:ring-2 focus-visible:ring-ring`                     |
| Hover only shifts alpha <15%                                        | Change a shape/edge channel (border, bg step, shadow)                  |
| Selected state styled with `success`                                | Selection → `primary`/`ring` + shape redundancy; `success` = outcomes  |

## System Map — Where Things Actually Live

Read the system in one pass before auditing (mapped during the 2026-11-22
Settings pixel audit):

| Concern | File | Notes |
| ------- | ---- | ----- |
| Palette tokens | `apps/web/palette.json` | Source of truth; regenerates `globals.css` (contract regen — commit e14975dc6 "token source of truth"). **Never hand-edit generated color rules in `globals.css`.** |
| Theme modes (Dark/Light/High Contrast) | `apps/web/lib/theme-storage.ts` | `THEME_PREVIEWS` powers the theme picker's tile mini-renders — by design they use this data, NOT live vars (only one mode's tokens are live at a time). Applied via `ThemeProvider.tsx` |
| strui button variants | `packages/strui/src/components/ui/button.tsx` | ghost/default/outline definitions |
| Selection/semantic tokens | `success`, `primary`, `accent`, `ring`, `border`, `muted` | see "selection ≠ outcome" below |
| Settings appearance UI | `apps/web/app/(app)/settings/settings-appearance.tsx` | theme tiles + palette section |
| Fixed overlays | FAB `apps/web/components/consciousness/ConsciousnessQuickActions.tsx` (z-9999); toasts `apps/web/features/chat/components/feedback/Toast.tsx` (z-300); bottom nav z-40 | each must declare its z-band contract vs the bottom nav |

## How to Audit — 3 Probes, All Three Required

1. **Alpha-corrected contrast scan** — composite the full background chain
   (element + ancestors over page bg) BEFORE computing the WCAG ratio.
   Ignoring alpha produced 2 false positives in the 2026-11-22 Settings audit
   (yellow-on-11%-yellow read as 1.0:1; actually fine).
2. **Hover before/after probe** — read computed style, dispatch a real hover
   event, read again. A `:hover` declared in source is NOT proof it shipped.
3. **Tab-ring check** — real `Tab`/`Shift+Tab` keypress, then read
   `getComputedStyle(document.activeElement)`. Focus-from-load does NOT
   trigger `:focus-visible`, so loading-state focus proves nothing.

## Canonical Findings (Settings audit, 2026-11-22)

1. **Declared-but-unemitted utility.** `button.tsx:15` ghost declares
   `hover:bg-accent/8`, but the emitted stylesheet has no `.bg-accent\/8`
   rule (emitted steps: /5 /10 /15 /20 /50 /80). Live proof: `:hover` matched
   `true` on the Settings Reset button while bg stayed `rgba(0,0,0,0)`.
   **Rule:** verify the utility exists in the emitted CSS (scan
   `document.styleSheets`) before trusting an opacity step; prefer steps
   known to be emitted.
2. **Overlay not nav-aware.** FAB `bottom-6` (24px) vs 56px bottom nav →
   32px overlap; at 390px viewport the FAB covers the last nav item's icon.
   **Rule:** any `position:fixed` bottom overlay must state its z-band
   contract vs the bottom nav; bottom offset ≥ nav height (or explicitly
   document covering it).
3. **Toast stack z-300 bottom-0** sits inside the nav band (full-width on
   mobile) → toasts cover nav items when present. Same rule as (2).
4. **Weak hover.** Palette outline buttons hover only shifts alpha 1→0.9 —
   imperceptible. **Rule:** hover must change a *shape/edge* channel (border,
   bg step, shadow), not just alpha <15%.

### Semantic-token rule: selection ≠ outcome

The theme tile selection uses `success` (`border-success bg-success/10` +
corner ribbon) — *selected is not an outcome*. **Rule:** selection →
`primary`/`ring` + shape redundancy (the ribbon shape + `aria-checked` are
what saved it); `success` is reserved for outcomes. Recorded as a rule only —
changing `settings-appearance.tsx` is a separate card.

### Audit baseline (passes worth recording)

0 contrast failures page-wide; equal tile geometry (223×106, 10px gaps);
focus rings present on tiles + palette (black 2px + yellow 4px double ring);
11px text floor held; no horizontal overflow; palette wraps cleanly at 390px;
a11y radiogroup/`aria-checked` solid.
