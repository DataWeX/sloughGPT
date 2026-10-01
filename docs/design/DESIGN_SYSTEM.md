# Noir Violet Design System — LOCKED

This is the **only** design system for sloughGPT. It does not change. All UI work must follow these rules exactly. Do not invent new colors, fonts, spacing, or patterns.

## Identity

**Name:** Noir Violet  
**Character:** Warm, sophisticated, technical. Rich violet primary with warm terracotta accents.  
**Mood:** Calm confidence. Not sterile, not playful. A tool that respects its user.

## The Two Non-Negotiables

Every page ships with the following two rules, and they outrank every other instruction in this document:

1. **Colors are never fixed in markup.** Zero hardcoded hex, zero stock Tailwind palette utilities, zero `hsl()` without a var reference. Every hue resolves from theme tokens — `rgb(var(--token))`, semantic utilities (`text-success`, `bg-warning`), or chart tokens (`var(--chart-N)`). This is what makes the theme switcher (light/dark × theme aura × palette) work: if markup pins a hue, the switcher becomes cosmetic and the page breaks the theme contract.
2. **Compose, don't stack.** Dashboards are built from the strui primitives below, never from raw `Card` stacks. If a layout would need more than two plain cards in a row with no hierarchy, the composition is wrong, not the page.

## Systematic & Functional UI

The dashboard re-skin (`consciousness/dashboard`) is the reference implementation of these principles:

- **One focal metric per page.** A page leads with a small `KpiGrid` of `StatCard`s hitting one hero number. Metrics never sit as five identically-weighted cards.
- **`KpiGrid` → `StatCard`** for any single-number callout. `StatCard` owns the stat type scale (its `text-xl` numeric value), the uppercase micro-label, and the trend tint (`text-success` / `text-destructive`).
- **Actions live in `PageContainer` `headerRight`**, not in an "Actions" card in the body.
- **`SectionHeader`** groups related content (metrics, quality/reflection, evolution, timeline). A `SectionHeader` + card beats a bare card for every section title.
- **`FoldSection`** (collapsible) for secondary controls — settings, snapshots, config. Keeping them collapsed distributes visual interest instead of packing the body with controls.
- **Live/streaming state is a `StatusDot`** (`tone="success" pulse showLabel`) — the visual "breathing" of live data — placed on a slim strip, not as a dotted badge inside a fifth card.
- **Chart series and progress bars use `var(--chart-N)`** or semantic tokens. If a series needs a distinct hue, it takes a token, never a hex constant.
- **Empty and loading states come from the components themselves** (`StatCard loading`, `PageContainer loadingGrid`, explicit "need at least 2 data points" callouts). No blank widgets, no endless spinners.

## Color Tokens

All colors are RGB triples used as `rgb(var(--token))` in CSS and `bg-token`, `text-token` in Tailwind.

> **Source of truth:** every value in this section is authored once in
> `packages/strui/tokens/palette.json` and _generated_ into the `@tokens:*` regions of
> `apps/web/app/globals.css` and `packages/strui/src/styles/globals.css`, into
> `apps/web/lib/theme-tokens.generated.ts` (switcher swatches),
> `apps/mobile/src/theme/palette.generated.ts`,
> `apps/mobile/src/theme/tamagui-themes.generated.ts` (tamagui theme overrides),
> `apps/web/lib/palette-showcase.generated.ts` (magazine showcase data), and the
> value tables in this document (the `<!-- @tokens:* -->` regions below — the
> Usage/Mood prose columns stay hand-written).
> To change a value: edit `palette.json`,
> run `node packages/strui/scripts/gen-tokens.mjs` (audit with `--check`), and commit the
> regenerated outputs — never hand-edit a generated file.
> `packages/strui/src/tokens/tokens.test.ts` enforces sync, tier order
> (base → palette → aura), aura scoping (`--primary`/`--ring` only), and totality.

### Light Mode

<!-- @tokens:light-tokens BEGIN -->

| Token                    | RGB           | Usage                                         |
| ------------------------ | ------------- | --------------------------------------------- |
| `--background`           | `248 246 252` | Page background (warm cream with violet tint) |
| `--foreground`           | `25 22 36`    | Primary text                                  |
| `--card`                 | `255 255 255` | Card/panel backgrounds                        |
| `--card-foreground`      | `25 22 36`    | Text on cards                                 |
| `--primary`              | `124 82 196`  | Buttons, links, active states                 |
| `--primary-foreground`   | `250 248 255` | Text on primary                               |
| `--secondary`            | `237 232 248` | Secondary backgrounds                         |
| `--secondary-foreground` | `42 37 55`    | Text on secondary                             |
| `--muted`                | `244 242 248` | Subtle backgrounds                            |
| `--muted-foreground`     | `130 122 150` | Captions, secondary text                      |
| `--accent`               | `236 145 95`  | Highlights, warnings, accents                 |
| `--accent-foreground`    | `250 248 255` | Text on accent                                |
| `--border`               | `228 224 242` | Borders, dividers                             |
| `--input`                | `228 224 242` | Input borders                                 |
| `--ring`                 | `124 82 196`  | Focus rings                                   |
| `--success`              | `52 176 125`  | Success states                                |
| `--warning`              | `236 168 60`  | Warning states                                |
| `--destructive`          | `220 80 90`   | Errors, destructive actions                   |

<!-- @tokens:light-tokens END -->

### Dark Mode

<!-- @tokens:dark-tokens BEGIN -->

| Token                    | RGB           | Usage                                  |
| ------------------------ | ------------- | -------------------------------------- |
| `--background`           | `17 15 24`    | Page background (deep charcoal-violet) |
| `--foreground`           | `238 234 248` | Primary text                           |
| `--card`                 | `28 25 38`    | Card/panel backgrounds                 |
| `--card-foreground`      | `238 234 248` | Text on cards                          |
| `--primary`              | `192 170 244` | Buttons, links, active states          |
| `--primary-foreground`   | `25 22 36`    | Text on primary                        |
| `--secondary`            | `50 44 68`    | Secondary backgrounds                  |
| `--secondary-foreground` | `238 234 248` | Text on secondary                      |
| `--muted`                | `38 34 52`    | Subtle backgrounds                     |
| `--muted-foreground`     | `150 140 172` | Captions, secondary text               |
| `--accent`               | `240 176 130` | Highlights, warnings, accents          |
| `--accent-foreground`    | `25 22 36`    | Text on accent                         |
| `--border`               | `52 46 72`    | Borders, dividers                      |
| `--input`                | `52 46 72`    | Input borders                          |
| `--ring`                 | `192 170 244` | Focus rings                            |
| `--success`              | `72 192 140`  | Success states                         |
| `--warning`              | `240 192 80`  | Warning states                         |
| `--destructive`          | `235 100 110` | Errors, destructive actions            |

<!-- @tokens:dark-tokens END -->

### Chart Colors

<!-- @tokens:chart-tokens BEGIN -->

| Token       | Light        | Dark          |
| ----------- | ------------ | ------------- |
| `--chart-1` | `124 82 196` | `192 170 244` |
| `--chart-2` | `52 176 125` | `72 192 140`  |
| `--chart-3` | `236 145 95` | `240 176 130` |
| `--chart-4` | `90 150 220` | `100 165 240` |
| `--chart-5` | `220 80 90`  | `235 100 110` |

<!-- @tokens:chart-tokens END -->

**Chart rule:** every chart series, legend swatch, progress bar, and per-item accent references a chart or semantic token — `stroke="var(--chart-1)"`, `backgroundColor: 'var(--chart-N)'`, `bg-success`/`bg-warning`/`bg-destructive`. Color constants at the top of a component (`const COLORS = { a: '#6366f1' }`) are forbidden; define them as `{ a: 'var(--chart-1)' }` so they follow theme and palette like everything else.

## Accent Auras

Each accent theme is a full aura — not just a primary color swap. Themes override `--primary`, `--ring`, `--secondary`, `--secondary-foreground`, `--muted`, `--border`, and `--input` to shift the entire UI atmosphere.

<!-- @tokens:accent-auras BEGIN -->

| Theme          | Aura       | Primary       | Mood                            |
| -------------- | ---------- | ------------- | ------------------------------- |
| `theme-blue`   | Periwinkle | `90 130 220`  | Calm ocean, trustworthy         |
| `theme-purple` | Lilac      | `155 108 214` | Ethereal lilac, mystical        |
| `theme-pink`   | Rose       | `218 130 170` | Soft rose, delicate             |
| `theme-red`    | Coral      | `230 120 130` | Warm coral, energetic           |
| `theme-orange` | Peach      | `236 155 90`  | Peachy bread crust, warm bakery |
| `theme-green`  | Mint       | `72 178 130`  | Fresh mint, natural             |
| `theme-teal`   | Dew        | `72 166 200`  | Cool dew, refreshing            |

<!-- @tokens:accent-auras END -->

Each theme also shifts `--secondary`, `--muted`, and `--border` with warm/cool tints to match the aura. Dark mode variants deepen these shifts.

## Typography

### Font Families

| Role    | Font                                     | Fallback                |
| ------- | ---------------------------------------- | ----------------------- |
| Body    | Rubik (`--font-rubik`)                   | system-ui, sans-serif   |
| Numeric | Lato (`--font-lato`)                     | system-ui, sans-serif   |
| Code    | JetBrains Mono (`--font-jetbrains-mono`) | ui-monospace, monospace |

### Type Scale

| Role          | Class                                          | Weight | Usage                        |
| ------------- | ---------------------------------------------- | ------ | ---------------------------- |
| Page title    | `text-2xl md:text-3xl font-semibold`           | 600    | Only in `AppRouteHeaderLead` |
| Section title | `text-base font-medium`                        | 500    | Card headers                 |
| Body          | `text-sm`                                      | 400    | Primary content              |
| Caption       | `text-xs text-muted-foreground`                | 400    | Timestamps, secondary info   |
| Label         | `text-xs font-medium uppercase tracking-wider` | 500    | Form labels                  |
| Badge         | `text-[10px] font-medium`                      | 500    | Status badges, tags          |

### Rules

- Never use `text-lg` in page body content
- Never use `text-2xl` outside `AppRouteHeaderLead`
- Never use `text-3xl` or larger in component content
- Body text is always `text-sm`
- Code blocks use `font-mono text-xs`

## Spacing

### Page Layout

| Token             | Class       | Value               |
| ----------------- | ----------- | ------------------- |
| Page wrapper      | `sl-page`   | `p-4 sm:p-6 md:p-8` |
| Max content width | `max-w-4xl` | 896px               |
| Between sections  | `space-y-4` | 16px                |
| Between cards     | `space-y-4` | 16px                |

### Component Spacing

| Context                 | Class                     |
| ----------------------- | ------------------------- |
| Card padding            | `p-4` or `p-6`            |
| Card header/content gap | `space-y-2`               |
| Button gap              | `gap-2`                   |
| Form field gap          | `space-y-2`               |
| Inline group            | `flex items-center gap-2` |
| Grid gap                | `gap-4`                   |

### Rules

- Never use arbitrary values like `px-[23]` or `py-[17]`
- Use standard Tailwind spacing: `p-1` through `p-6`, `gap-1` through `gap-6`
- Page wrapper is always `sl-page mx-auto max-w-4xl`

## Border Radius

| Token        | Value  | Usage                      |
| ------------ | ------ | -------------------------- |
| `--radius`   | `6px`  | Default for all components |
| `rounded-sm` | `2px`  | Subtle rounding            |
| `rounded-lg` | `10px` | Modals, large panels       |
| `rounded-xl` | `14px` | Feature cards              |

## Shadows

| Token       | Usage                            |
| ----------- | -------------------------------- |
| `shadow-sm` | Subtle elevation (cards at rest) |
| `shadow-md` | Hover states, dropdowns          |
| `shadow-lg` | Modals, popovers                 |
| `shadow-xl` | Command palette, floating panels |

## Component Patterns

### Cards

```tsx
import { Card, CardHeader, CardTitle, CardContent } from '@anthropic/strui/card'

;<Card>
  <CardHeader>
    <CardTitle className="text-base">Section Title</CardTitle>
  </CardHeader>
  <CardContent>
    <p className="text-sm">Content here</p>
  </CardContent>
</Card>
```

### Buttons

```tsx
import { Button } from '@anthropic/strui/button'

// Primary action
<Button>Save Changes</Button>

// Secondary action
<Button variant="secondary">Cancel</Button>

// Destructive action
<Button variant="destructive">Delete</Button>

// Ghost/minimal
<Button variant="ghost">Learn More</Button>
```

### Forms

```tsx
import { Input } from '@anthropic/strui/input'
import { Label } from '@anthropic/strui/label'

;<div className="space-y-2">
  <Label htmlFor="name">Name</Label>
  <Input id="name" placeholder="Enter name" />
</div>
```

### Badges

```tsx
import { Badge } from '@anthropic/strui/badge'

<Badge variant="default">Active</Badge>
<Badge variant="secondary">Draft</Badge>
<Badge variant="destructive">Error</Badge>
```

### Page Template

```tsx
<div className="sl-page mx-auto max-w-4xl">
  <AppRouteHeader left={<AppRouteHeaderLead title="Page Title" subtitle="Description" />} />
  <div className="space-y-4">
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Section</CardTitle>
      </CardHeader>
      <CardContent>{/* Content */}</CardContent>
    </Card>
  </div>
</div>
```

## Interactive States

### Hover

Every clickable element must have a hover state:

| Element          | Hover Class                               |
| ---------------- | ----------------------------------------- |
| Button           | `hover:bg-primary/90`                     |
| Card (clickable) | `hover:border-primary/50 hover:shadow-md` |
| Link             | `hover:text-primary/80`                   |
| List item        | `hover:bg-muted`                          |
| Icon button      | `hover:bg-muted hover:text-foreground`    |

### Focus

All interactive elements must have visible focus:

```tsx
className = 'focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2'
```

### Disabled

Disabled elements use reduced opacity:

```tsx
className = 'opacity-40 pointer-events-none'
```

## Animation

- Use `transition-smooth` (`cubic-bezier(0.4, 0, 0.2, 1)`) for most transitions
- Duration: `duration-150` for micro-interactions, `duration-200` for page transitions
- Respect `prefers-reduced-motion` — no animations for users who request it
- No bouncing, no spinning (except loading spinners), no flashy effects

## Forbidden Patterns

| Never                                                                                         | Use Instead                                                                         |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `#hex` colors                                                                                 | `rgb(var(--token))` or Tailwind classes                                             |
| `text-gray-500`                                                                               | `text-muted-foreground`                                                             |
| `bg-white`                                                                                    | `bg-card`                                                                           |
| `border-gray-200`                                                                             | `border-border`                                                                     |
| `text-lg` in body                                                                             | `text-sm` or `text-base`                                                            |
| `px-8 py-6` on page                                                                           | `sl-page` class                                                                     |
| Inline `style={{}}`                                                                           | Tailwind classes                                                                    |
| `console.log` in production                                                                   | Remove before commit                                                                |
| Custom color variables                                                                        | Use existing tokens                                                                 |
| New font families                                                                             | Use Rubik, Lato, or JetBrains Mono                                                  |
| `rounded-full` on cards                                                                       | Use `rounded` or `rounded-lg`                                                       |
| Animations without motion check                                                               | Add `motion-reduce:` variants                                                       |
| Stock Tailwind palette (`text-green-500`, `bg-red-400`, `border-blue-500`, `text-indigo-600`) | Semantic tokens (`text-success`, `bg-destructive`, `border-border`, `text-primary`) |
| Hex in chart series / progress bars (`#6366f1`, `#22c55e`)                                    | `var(--chart-N)` / `bg-success` / `bg-warning` / `bg-destructive`                   |
| Loud AI-default hero gradients (vivid purple→indigo at full opacity)                          | Token-tinted ambient washes only (below)                                            |
| "Actions" card in the page body                                                               | `PageContainer headerRight`                                                         |
| Five identical stat cards (`Card` + `text-2xl`)                                               | `KpiGrid` + `StatCard`                                                              |
| Stat numbers at `text-2xl` in body                                                            | `StatCard` value (owns its own scale)                                               |
| Live/status conveyed as raw text + `bg-green-400` dot                                         | `StatusDot` (`tone="success" pulse`)                                                |
| Secondary controls stacked as cards                                                           | `FoldSection` (collapsible)                                                         |
| A row/stack of more than two flat cards with no hierarchy                                     | `SectionHeader`, `KpiGrid`, or rethink the layout                                   |

## AI Agent Rules

1. **This design system is locked.** Do not propose changes to colors, fonts, spacing, or component patterns.
2. **Do not use the frontend-design skill** for sloughGPT. It is for new projects only.
3. **Always use existing tokens.** Never invent new colors or spacing values.
4. **Always use strui components.** Do not build custom UI primitives.
5. **Always follow the type scale.** No arbitrary font sizes.
6. **Always add hover states.** No clickable elements without feedback.
7. **Always use focus rings.** Accessibility is not optional.
8. **Test in both light and dark mode.** Every component must work in both.
9. **Never fix a hue in markup.** No hex, no stock Tailwind palette, no `hsl()` without a var. All color resolves through tokens, semantic utilities, or `var(--chart-N)`.
10. **Compose dashboards from primitives.** `KpiGrid`/`StatCard` for metrics, `SectionHeader` for grouping, `FoldSection` for secondary controls, `StatusDot` for live state, actions in `headerRight`. Raw card stacks are the exception, not the default.
11. **Verify against the AI-slop scanner** below or run the `design-system-enforcement` skill before committing UI changes.

## Dispatch the AI Slop (self-check before any UI commit)

A page has AI-slop fingerprints if any of these are true — fix them before merge:

1. It uses stock Tailwind color utilities (`text-green-500`, `bg-blue-400`, `border-amber-500`, …) or hex constants.
2. It stacks three or more identical cards with no `KpiGrid`/`StatCard`/`SectionHeader` hierarchy.
3. Stat numbers rendered at `text-lg`/`text-2xl` in the body instead of via `StatCard`.
4. Chart series or progress bars print a fixed hue instead of `var(--chart-N)`.
5. Live/status indicators are text + floating colored dot inside a card.
6. It reads like "any SaaS product" — a generic dashboard the viewer can't tell belongs to Noir Violet (no StatDots, no loss-curve motif, no personality).

### The Wine Wash is Signature, Not Slop

The token-tinted ambient gradient is sloughGPT's personality and is **explicitly protected** — do not "clean it up":

- `ActiveModelBanner` on the home page: `bg-gradient-to-br from-primary/[0.04] via-transparent to-accent/[0.03]` — the violet→peach "wine" header, and its skeleton (`ActiveModelBannerSkeleton`) uses the same wash so the loading state keeps it.
- `.sl-sidebar-surface` and `.sl-mobile-header` radial washes in `globals.css` (primary/accent at 7–10% opacity).

These are allowed because they are **token-tinted and near-transparent**. They must never become vivid or hardcoded. If an audit flags them along a stock-palette run, they are the exception — the AI-slop ban targets loud stock-hue gradients, never this wash.

The audit command that found the 367-instance outbreak and every offender in the `consciousness/*` family:

```bash
# Stock Tailwind palette (AI-slop rainbow) — all must be zero
grep -rEo '(bg|text|border|ring|from|to|via|divide|decoration|accent|caret|fill|stroke|shadow|outline)-(indigo|fuchsia|cyan|sky|emerald|lime|amber|teal|rose|pink|red|yellow|green|blue|orange|purple)-(50|100|200|300|400|500|600|700|800|900|950)' \
  apps/web/app/\(app\)/ --include=page.tsx | wc -l

# Hex in components
grep -rn '#[0-9a-fA-F]\{3,8\}' apps/web/ --include='*.tsx' | grep -v '.test.\|stories.\|node_modules'

# Hex chart/list constants at top of a component (forbidden)
grep -rn "': '#" apps/web/components apps/web/app --include='*.tsx' | grep -v '.test.'
```
