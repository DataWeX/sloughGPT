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

### Light Mode

| Token | RGB | Usage |
|-------|-----|-------|
| `--background` | `248 246 252` | Page background (warm cream with violet tint) |
| `--foreground` | `25 22 36` | Primary text |
| `--card` | `255 255 255` | Card/panel backgrounds |
| `--card-foreground` | `25 22 36` | Text on cards |
| `--primary` | `124 82 196` | Buttons, links, active states |
| `--primary-foreground` | `250 248 255` | Text on primary |
| `--secondary` | `237 232 248` | Secondary backgrounds |
| `--secondary-foreground` | `42 37 55` | Text on secondary |
| `--muted` | `244 242 248` | Subtle backgrounds |
| `--muted-foreground` | `130 122 150` | Captions, secondary text |
| `--accent` | `236 145 95` | Highlights, warnings, accents |
| `--accent-foreground` | `250 248 255` | Text on accent |
| `--border` | `228 224 242` | Borders, dividers |
| `--input` | `228 224 242` | Input borders |
| `--ring` | `124 82 196` | Focus rings |
| `--success` | `52 176 125` | Success states |
| `--warning` | `236 168 60` | Warning states |
| `--destructive` | `220 80 90` | Errors, destructive actions |

### Dark Mode

| Token | RGB | Usage |
|-------|-----|-------|
| `--background` | `17 15 24` | Page background (deep charcoal-violet) |
| `--foreground` | `238 234 248` | Primary text |
| `--card` | `28 25 38` | Card/panel backgrounds |
| `--card-foreground` | `238 234 248` | Text on cards |
| `--primary` | `192 170 244` | Buttons, links, active states |
| `--primary-foreground` | `25 22 36` | Text on primary |
| `--secondary` | `50 44 68` | Secondary backgrounds |
| `--secondary-foreground` | `238 234 248` | Text on secondary |
| `--muted` | `38 34 52` | Subtle backgrounds |
| `--muted-foreground` | `150 140 172` | Captions, secondary text |
| `--accent` | `240 176 130` | Highlights, warnings, accents |
| `--accent-foreground` | `25 22 36` | Text on accent |
| `--border` | `52 46 72` | Borders, dividers |
| `--input` | `52 46 72` | Input borders |
| `--ring` | `192 170 244` | Focus rings |
| `--success` | `72 192 140` | Success states |
| `--warning` | `240 192 80` | Warning states |
| `--destructive` | `235 100 110` | Errors, destructive actions |

### Chart Colors

| Token | Light | Dark |
|-------|-------|------|
| `--chart-1` | `124 82 196` | `192 170 244` |
| `--chart-2` | `52 176 125` | `72 192 140` |
| `--chart-3` | `236 145 95` | `240 176 130` |
| `--chart-4` | `90 150 220` | `100 165 240` |
| `--chart-5` | `220 80 90` | `235 100 110` |

**Chart rule:** every chart series, legend swatch, progress bar, and per-item accent references a chart or semantic token — `stroke="var(--chart-1)"`, `backgroundColor: 'var(--chart-N)'`, `bg-success`/`bg-warning`/`bg-destructive`. Color constants at the top of a component (`const COLORS = { a: '#6366f1' }`) are forbidden; define them as `{ a: 'var(--chart-1)' }` so they follow theme and palette like everything else.

## Accent Auras

Each accent theme is a full aura — not just a primary color swap. Themes override `--primary`, `--ring`, `--secondary`, `--secondary-foreground`, `--muted`, `--border`, and `--input` to shift the entire UI atmosphere.

| Theme | Aura | Primary | Mood |
|-------|------|---------|------|
| `theme-blue` | Periwinkle | `90 130 220` | Calm ocean, trustworthy |
| `theme-purple` | Lilac | `155 108 214` | Ethereal lilac, mystical |
| `theme-pink` | Rose | `218 130 170` | Soft rose, delicate |
| `theme-red` | Coral | `230 120 130` | Warm coral, energetic |
| `theme-orange` | Peach | `236 155 90` | Peachy bread crust, warm bakery |
| `theme-green` | Mint | `72 178 130` | Fresh mint, natural |
| `theme-teal` | Dew | `72 166 200` | Cool dew, refreshing |

Each theme also shifts `--secondary`, `--muted`, and `--border` with warm/cool tints to match the aura. Dark mode variants deepen these shifts.

## Typography

### Font Families

Two families ship: **Outfit** (body, headings, numerals) and **JetBrains Mono** (code). No third family is loaded in the app.

| Role | Font | CSS variable | Fallback |
|------|------|--------------|----------|
| Body / headings (`font-sans`) | Outfit — `public/fonts/outfit-latin.woff2` | `--font-rubik` — **legacy name; the family is Outfit** | system-ui, sans-serif |
| Numeric (`font-numeric`, `tabular-nums`) | Outfit — same face as body; there is no separate numeral font | `--font-rubik` | system-ui, sans-serif |
| Code (`font-mono`) | JetBrains Mono — `public/fonts/jetbrains-latin.woff2` | `--font-jetbrains-mono` | ui-monospace, monospace |

- Outfit is loaded with `next/font/local` in `apps/web/app/layout.tsx`; JetBrains Mono also has a raw `@font-face` in `globals.css`.
- **`--font-rubik` is a rename TODO, not a font.** The variable resolves to Outfit. Do not rename it in passing — consumers are `layout.tsx`, `globals.css`, `tailwind.config.js`, `magazine/page.tsx`, and `vite/index-metadata.test.ts`. Until the rename lands, read it as "the body font".
- **Lato was never loaded.** No `--font-lato`, no Lato file, no `@font-face` — the old Numeric row was doc drift, not the system.
- `apps/web/lib/fonts.ts` lists a body-font switcher (Inter / Outfit / Space Grotesk / Plus Jakarta) but ships no font files for it; Outfit is the only body face in use.
- Mobile maps the same two families through `apps/mobile/tamagui.config.ts` — see **Mobile (React Native / Tamagui)** below.

### Type Scale

| Role | Class | Weight | Usage |
|------|-------|--------|-------|
| Page title | `text-2xl md:text-3xl font-semibold` | 600 | Only in `AppRouteHeaderLead` |
| Section title | `text-base font-medium` | 500 | Card headers |
| Body | `text-sm` | 400 | Primary content |
| Caption | `text-xs text-muted-foreground` | 400 | Timestamps, secondary info |
| Label | `text-xs font-medium uppercase tracking-wider` | 500 | Form labels |
| Badge | `text-[10px] font-medium` | 500 | Status badges, tags |

### Rules

- Never use `text-lg` in page body content
- Never use `text-2xl` outside `AppRouteHeaderLead`
- Never use `text-3xl` or larger in component content
- Body text is always `text-sm`
- Code blocks use `font-mono text-xs`

## Spacing

### Page Layout

| Token | Class | Value |
|-------|-------|-------|
| Page wrapper | `sl-page` | `p-4 sm:p-6 md:p-8` |
| Max content width | `max-w-4xl` | 896px |
| Between sections | `space-y-4` | 16px |
| Between cards | `space-y-4` | 16px |

### Component Spacing

| Context | Class |
|---------|-------|
| Card padding | `p-4` or `p-6` |
| Card header/content gap | `space-y-2` |
| Button gap | `gap-2` |
| Form field gap | `space-y-2` |
| Inline group | `flex items-center gap-2` |
| Grid gap | `gap-4` |

### Rules

- Never use arbitrary values like `px-[23]` or `py-[17]`
- Use standard Tailwind spacing: `p-1` through `p-6`, `gap-1` through `gap-6`
- Page wrapper is always `sl-page mx-auto max-w-4xl`

## Border Radius

| Token | Value | Usage |
|-------|-------|-------|
| `--radius` | `6px` | Default for all components |
| `rounded-sm` | `2px` | Subtle rounding |
| `rounded-lg` | `10px` | Modals, large panels |
| `rounded-xl` | `14px` | Feature cards |
| `rounded-2xl` | `16px` | Chat composer card, floating input surfaces |
| `rounded-full` | `9999px` | Send/mic buttons, avatars, status dots — never on cards |

## Shadows

| Token | Usage |
|-------|-------|
| `shadow-sm` | Subtle elevation (cards at rest) |
| `shadow-md` | Hover states, dropdowns |
| `shadow-lg` | Modals, popovers |
| `shadow-xl` | Command palette, floating panels |

## Component Patterns

### Cards

```tsx
import { Card, CardHeader, CardTitle, CardContent } from '@anthropic/strui/card'

<Card>
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

<div className="space-y-2">
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

### The Log Entry — signature layout primitive

sloughGPT's visual signature is **one strong layout primitive, repeated**:
the annotated log entry. Every list of things a user reads turn-by-turn —
chat transcripts, the kanban board, later anything else — composes through
the same three-zone anatomy, so the silhouette (gutter + flowing text +
metadata margin) is recognizably sloughGPT even shrunk to black and white.

```
┌ gutter ┐ content (document-like)              ┐ margin (mono metadata) ┐
│  07    │ The words are the interface.         │ role · model · 2.1 t/s │
│        │ Assistant prose has NO bubble chrome.│ ● live · 148 tok       │
```

- **Gutter** (`font-mono text-[10px] text-muted-foreground/50`): stable
  position marker — turn number in chat, hash/id prefix on board entries.
- **Content**: assistant turns are flowing prose with no card or bubble;
  user turns are the only contained element (right-aligned tinted card in
  chat). Board entries sit in their column like ledger lines, not floating
  rounded cards.
- **Margin** (`font-mono text-[10px] text-muted-foreground/60`): metadata as
  first-class typography beside the content it describes (Tufte sidenote
  model) — per-turn tok/s, token count, latency, model; card priority,
  assignee, due date. Live/in-flight state is a `StatusDot tone="primary"
  pulse`; WIP overflow is `StatusDot tone="destructive" pulse`.

Reference implementations: `features/chat/components/messages/MessageBubble.tsx`
(chat turn), `components/planner/CardItem.tsx` + `Tray.tsx` (board entry +
column). Per-turn chat metrics arrive via `streamChatResponse`'s `onMeta`
(`{tokens, elapsedMs}`) and live on `ChatMessage.metrics`.

Rules for new surfaces adopting the primitive:

1. Three zones, always — never re-introduce a bubble/bubble pair.
2. Metadata goes in the margin, never as badges stacked under the content.
3. One margin annotation per fact; no icon+label pairs (icon-only with
   `aria-label`/`title` is fine).
4. Gutter is `aria-hidden`; the margin carries the accessible text.

### Page Template

```tsx
<div className="sl-page mx-auto max-w-4xl">
  <AppRouteHeader
    left={<AppRouteHeaderLead title="Page Title" subtitle="Description" />}
  />
  <div className="space-y-4">
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Section</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Content */}
      </CardContent>
    </Card>
  </div>
</div>
```

### Chat Composer / Chat Screen

Codified from the archived references `docs/design/references/chat-composer-reference-1.png` and `-2.png`. The composer is one floating card with **two rows**: message field on top, controls underneath.

```text
┌────────────────────────────────────────┐
│  message input                         │   row 1 — Body type, status dot top-right
│                                        │
│  [ + ]             Model [v]      [ ^ ]│   row 2 — controls on their own line
└────────────────────────────────────────┘
```

| Part | Rule |
|------|------|
| Shell | One card: `rounded-2xl border border-border/50 bg-card px-4 py-3 shadow-sm`. On focus: `focus-within:border-primary/40 focus-within:shadow-md focus-within:ring-1 focus-within:ring-primary/10`. Reference impl: `features/chat/components/input/ChatInputRow.tsx`. |
| Row 1 — field | Multiline message input in Body type, placeholder `text-muted-foreground`. Grows to a max height, scrolls internally. |
| Row 2 — controls | A separate row **below** the field — controls never sit inline beside the first line of typed text. |
| `+` affordance | Bottom-**left**: `h-9 w-9` ghost icon button (`hover:bg-muted`), `aria-label` naming the action. |
| Model picker | Bottom-**right**, immediately left of send: text button in Label/Caption type, `text-muted-foreground` → foreground on hover, chevron, shows the active model. |
| Send button | Bottom-**right**: circular `h-9 w-9 rounded-full bg-primary`, `aria-label="Send message"`. Streaming → Stop in the same circle (`bg-destructive`). Empty → mic/neutral state. Fixed footprint in every state — no layout shift. |
| Gutter | Stays inside the page gutter (`px-4 sm:px-6`) and the `max-w-4xl` thread column; the thread gets bottom padding ≥ composer height + safe-area so the last message is never covered. |
| Hint strip (optional) | Below row 2, Badge type (`text-[10px] text-muted-foreground/40`), `aria-hidden` — kbd hints, token count. |

**Message-list type roles**

| Element | Role |
|---------|------|
| Message text | Body (`text-sm`; RN 15/22) |
| Timestamps, model/meta lines, queued count | Caption (`text-xs text-muted-foreground`; RN 12/16) |
| Role labels, mode chips | Label (`text-xs font-medium uppercase tracking-wider`) |
| Status badges, match counters | Badge (`text-[10px]`) |

**Empty / error / offline rows**

- **Empty thread** → centered empty state: icon + context + one next action, composer still active below. Never a blank screen.
- **Error** → inline banner row between header and thread: dot + message + Retry + dismiss (web `SystemBanner`; mobile ChatScreen error row). Blocking errors are never toast-only.
- **Offline** → thin status strip (`ConnectionStatusBar` / `SystemBanner`) with a Caption-type queued count and Retry. Sends queue locally; the composer stays usable.
- **Streaming** → send morphs to Stop inside the same fixed-size circle; the field does not resize.

Existing implementations: web `apps/web/features/chat/components/input/` (`ChatInputRow`, `ChatInputField`, `ChatInputAccessories`, `ChatSendButton`), mobile `apps/mobile/src/components/ChatInput.tsx`.

## Interactive States

### Hover

Every clickable element must have a hover state:

| Element | Hover Class |
|---------|-------------|
| Button | `hover:bg-primary/90` |
| Card (clickable) | `hover:border-primary/50 hover:shadow-md` |
| Link | `hover:text-primary/80` |
| List item | `hover:bg-muted` |
| Icon button | `hover:bg-muted hover:text-foreground` |

### Focus

All interactive elements must have visible focus:

```tsx
className="focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
```

### Disabled

Disabled elements use reduced opacity:

```tsx
className="opacity-40 pointer-events-none"
```

## Animation

- Use `transition-smooth` (`cubic-bezier(0.4, 0, 0.2, 1)`) for most transitions
- Duration: `duration-150` for micro-interactions, `duration-200` for page transitions
- Respect `prefers-reduced-motion` — no animations for users who request it
- No bouncing, no spinning (except loading spinners), no flashy effects

## Mobile (React Native / Tamagui)

`apps/mobile` runs the same locked system through Tamagui. This section is the mapping, not a second system — every value below already exists on web; nothing here is invented.

### Font Mapping

| Role | Mobile | Where |
|------|--------|-------|
| Body / headings | Outfit | `apps/mobile/tamagui.config.ts` → `fonts.body`, `fonts.heading` |
| Code | JetBrains Mono (`JetBrainsMono-Regular`) | `fonts.mono` |
| Weights | 400 / 500 / 600 / 700 | weight maps in `tamagui.config.ts` |

Same two families as web, no others. No platform-switched font, no newly bundled family.

### Type Scale (RN)

Web class → RN `fontSize` / `lineHeight` (px):

| Role | Web class | RN fontSize / lineHeight | Weight |
|------|-----------|--------------------------|--------|
| Page title | `text-2xl md:text-3xl` | 24 / 30 | 600 |
| Section title | `text-base` | 16 / 22 | 500 |
| Body | `text-sm` | 15 / 22 | 400 |
| Caption | `text-xs` | 12 / 16 | 400 |
| Label | `text-xs font-medium uppercase tracking-wider` | 12 / 16, uppercase, tracking ≥ 0.5 | 500 |
| Badge | `text-[10px] font-medium` | 11 / 15 | 500 |

- Body is 15/22 on mobile (1px over web `text-sm`, reading distance) — what `MessageBubble` and `ChatInput` already ship.
- The five chat files carry **ten distinct `fontSize` literals (9–24) for six roles**. Each maps to a row above: 9–10 → Badge, 11 → Caption/Badge, 12–13 → Caption, 14–15 → Body, 16 → Section title, 18/20 → icon glyphs only (never body text), 24 → Page title.

### Spacing Grid

All spacing is a multiple of 4 (px): **4 · 8 · 12 · 16 · 20 · 24 · 32** — 48 only for full-bleed empty states. `2` is the one half-step, for hairline gaps and dots.

| Context | Value |
|---------|-------|
| Icon-to-label micro gap | 4 |
| Inline gap, tight stack | 8 |
| Card / composer internal padding | 12–16 |
| Screen gutter, between sections | 16 |
| Sheet padding, empty-state gutter | 20–32 |

`MessageBubble`'s local `S = {xs: 4, sm: 8, md: 12, lg: 16, xl: 20}` is already this grid; the shared scale adds 24 and 32. Off-grid literals (`paddingVertical: 7`, `paddingHorizontal: 14`, `marginHorizontal: 6`) are drift.

### Radius

| Token | Value | Web class | Use |
|-------|-------|-----------|-----|
| `xs` | 2 | `rounded-sm` | Hairlines, subtle rounding |
| `sm` | 6 | `rounded` (`--radius`) | Inputs, chips |
| `md` | 10 | `rounded-lg` | Buttons, list tiles, icon buttons |
| `lg` | 14 | `rounded-xl` | Cards, sheets |
| `xl` | 16 | `rounded-2xl` | Chat composer, floating panels |
| `full` | 9999 | `rounded-full` | Send/mic buttons, avatars, dots — never cards |

`MessageBubble`'s local `R = {sm: 6, md: 10, lg: 16, xl: 20, full: 9999}` predates this table: 6/10 land on `sm`/`md`, 16 maps to `xl`, 20 is off-scale. The five chat files use **sixteen distinct radius literals (2–28 plus 999/9999)** — all of them collapse into this table.

### Color Pipeline

- **Source of truth:** `packages/strui/tokens/palette.json` — the same values as the CSS variables in **Color Tokens** above. Regenerate the web and mobile outputs with `node packages/strui/scripts/gen-tokens.mjs` (`--check` audits without writing); never hand-edit a generated file.
- Mobile reaches the palette through `useColors()` (`src/theme/colors.ts`, backed by `palette.generated.ts`) and the theme overrides in `tamagui.config.ts`. Where the generator has not landed yet, those are hand-maintained mirrors of the same values — divergence from `palette.json` is a bug either way.
- A palette change edits `palette.json` and regenerates: web and mobile move together in one change.
- Components never inline a hex: Tamagui theme tokens (`$background`, `$color`, `$color10`) or `useColors()` — the mobile form of Non-Negotiable #1.

### No Magic Numbers

**Hard rule: no raw spacing, radius, or type numbers in components.** Every such value comes from the central scales module — `apps/mobile/src/theme/scales.ts` (created in the implementation phase):

```tsx
// scales.ts — MessageBubble.tsx's local constants, promoted to one shared module
export const S = {xs: 4, sm: 8, md: 12, lg: 16, xl: 20, xxl: 24, xxxl: 32} as const // 4pt grid
export const R = {xs: 2, sm: 6, md: 10, lg: 14, xl: 16, full: 9999} as const // radius

export const T = { // type styles
  page: {fontSize: 24, lineHeight: 30, fontWeight: '600'},
  section: {fontSize: 16, lineHeight: 22, fontWeight: '500'},
  body: {fontSize: 15, lineHeight: 22},
  caption: {fontSize: 12, lineHeight: 16},
  label: {fontSize: 12, lineHeight: 16, letterSpacing: 0.6},
  badge: {fontSize: 11, lineHeight: 15, fontWeight: '500'},
} as const
```

- Literal exceptions: icon/glyph sizes (14/16/18/20) and 1–2px hairlines/borders. Nothing else.
- Colors are exempt from `scales.ts` — they come from theme tokens or `useColors()` only.
- The five chat files (`ChatScreen`, `ChatInput`, `ChatModeBar`, `SidebarDrawer`, `ChatDrawer`) are the first migration targets; `MessageBubble` already follows the pattern locally.

## Forbidden Patterns

| Never | Use Instead |
|-------|-------------|
| `#hex` colors | `rgb(var(--token))` or Tailwind classes |
| `text-gray-500` | `text-muted-foreground` |
| `bg-white` | `bg-card` |
| `border-gray-200` | `border-border` |
| `text-lg` in body | `text-sm` or `text-base` |
| `px-8 py-6` on page | `sl-page` class |
| Inline `style={{}}` | Tailwind classes |
| `console.log` in production | Remove before commit |
| Custom color variables | Use existing tokens |
| New font families | Use Outfit or JetBrains Mono |
| `rounded-full` on cards | Use `rounded` or `rounded-lg` |
| Animations without motion check | Add `motion-reduce:` variants |
| Stock Tailwind palette (`text-green-500`, `bg-red-400`, `border-blue-500`, `text-indigo-600`) | Semantic tokens (`text-success`, `bg-destructive`, `border-border`, `text-primary`) |
| Hex in chart series / progress bars (`#6366f1`, `#22c55e`) | `var(--chart-N)` / `bg-success` / `bg-warning` / `bg-destructive` |
| Loud AI-default hero gradients (vivid purple→indigo at full opacity) | Token-tinted ambient washes only (below) |
| "Actions" card in the page body | `PageContainer headerRight` |
| Five identical stat cards (`Card` + `text-2xl`) | `KpiGrid` + `StatCard` |
| Stat numbers at `text-2xl` in body | `StatCard` value (owns its own scale) |
| Live/status conveyed as raw text + `bg-green-400` dot | `StatusDot` (`tone="success" pulse`) |
| Secondary controls stacked as cards | `FoldSection` (collapsible) |
| A row/stack of more than two flat cards with no hierarchy | `SectionHeader`, `KpiGrid`, or rethink the layout |
| Chat composer controls inline beside the text field | Two-row composer (**Chat Composer / Chat Screen**) |
| Raw spacing/radius/fontSize literals in RN components (`paddingVertical: 7`, `borderRadius: 9`, `fontSize: 13`) | Scales from `src/theme/scales.ts` (**Mobile (React Native / Tamagui)** section) |

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

# Chat transcript must stay bubble-free (the log-entry primitive)
grep -rnE 'rounded-(br|bl)-md|chat-bubble' apps/web/features/chat/components/messages/ --include='*.tsx' | grep -v '.test.'   # must be zero

# Planner must not pin hues (non-negotiable #1)
grep -rn 'hsl(' apps/web/components/planner --include='*.tsx' | grep -v '.test.'   # must be zero
```
