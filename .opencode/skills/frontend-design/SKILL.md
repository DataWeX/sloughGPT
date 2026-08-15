---
name: frontend-design
description: Use when building or reshaping UI components, pages, or layouts in sloughGPT. Enforces the Noir Violet design system, prevents AI-default patterns, and ensures distinctive visual identity. Apply before any CSS, component, or layout work.
---

# Frontend Design — sloughGPT

You are the design lead for sloughGPT, a local-first AI training platform with a "Noir Violet" visual identity. Your job is to make every UI decision deliberately — never because it's the easy default.

## The Design System You Inherit

### Palette — Noir Violet (RGB triples, never hex)

| Role | Light | Dark | CSS variable |
|------|-------|------|-------------|
| Primary | `124 82 196` (violet) | `192 170 244` (lilac) | `--primary` |
| Accent | `236 145 95` (terracotta) | `240 176 130` (peach) | `--accent` |
| Background | `248 246 252` (warm cream) | `17 15 24` (charcoal) | `--background` |
| Card | `255 255 255` | `28 25 38` | `--card` |
| Border | `228 224 242` | `52 46 72` | `--border` |

**Rules:**
- ALL colors via CSS custom properties — `rgb(var(--primary))`, never `#7c52c4` or `hsl(...)`
- Use `color-mix()` for opacity — `color-mix(in srgb, rgb(var(--primary)) 13%, transparent)`
- Theme accent chips: `.theme-blue`, `.theme-purple`, `.theme-pink`, etc. override `--primary`
- Shadow tokens: `--shadow-sm` through `--shadow-xl` in `globals.css`

### Typography

- **Display**: Outfit (400/500/600/700) — served locally from `/public/fonts/`
- **Mono**: JetBrains Mono (400/500) — code, technical values
- Body: `text-sm` (14px). Section titles: `text-base font-medium`. Page titles: `AppRouteHeaderLead`.
- Never use `text-lg` or `text-2xl` in page body.

### Layout

```
sl-page mx-auto max-w-4xl space-y-4
  AppRouteHeader(left=<AppRouteHeaderLead />)
  Card > CardHeader > CardTitle text-base + CardContent
```

## What Makes This UI Feel "Boring" — And How to Fix It

### Problem 1: Card Soup
Every page is 6-12 cards stacked vertically with identical padding and no visual hierarchy. Cards are the only layout primitive.

**Fix:** Break the card grid intentionally.
- Use `KpiGrid` for 2-4 stats (2×2 grid, tight gaps)
- Use full-width cards only for primary content
- Add `StatCard` for single-number callouts
- Let some content breathe outside cards (text blocks, inline lists)
- Use `FoldSection` for collapsible secondary content

### Problem 2: Everything Looks the Same Weight
Every card has `CardHeader > CardTitle` at `text-base font-medium`. No visual hierarchy — the eye has nowhere to land.

**Fix:** Create a clear visual hierarchy.
- One hero element per page (the most important action or stat)
- Section dividers with `SectionHeader` — not just another card
- Status badges (`bg-success/10 text-success`) for quick scanning
- Muted secondary info: `text-xs text-muted-foreground`

### Problem 3: No Personality
The UI reads like a well-organized dashboard but has no character. It could be any SaaS product.

**Fix:** Add sloughGPT's specific personality.
- Use the training loss curve SVG as a recurring visual motif (not just on home)
- Status dots (`StatusDot`) for live system state — gives the interface a "breathing" quality
- Soul personality traits displayed as colored chips, not just text
- Loading states with `LoadingDots` instead of generic skeleton pulses

### Problem 4: Dense Headers, Empty Bodies
Pages like Settings and Models have packed headers with soul pills, model selectors, and checkpoint badges — but the body is just cards with sliders.

**Fix:** Distribute visual interest.
- Move secondary controls into `FoldSection` (collapsible)
- Use the sidebar `ChatMoreMenu` pattern for overflow options
- Let body content have more whitespace

### Problem 5: No Motion
No page transitions, no hover feedback, no loading reveals. The UI appears and disappears.

**Fix:** Add deliberate motion.
- `prefers-reduced-motion` respected — use `transition-all duration-200` on interactive elements
- Cards: `hover:-translate-y-0.5 hover:shadow-md` for clickable cards
- Buttons: `active:scale-[0.98]` for press feedback
- Page content: `animate-in fade-in` on mount (subtle)
- Toast notifications: `sl-toast-in` animation (already exists)

## Anti-Patterns to Avoid

| Don't | Why | Do instead |
|-------|-----|-----------|
| Add more cards | Card soup | Use KpiGrid, StatCard, SectionHeader, inline content |
| Use `px-8 py-6` arbitrary spacing | Breaks spacing system | Use `sl-page` padding, `space-y-4`, `gap-*` tokens |
| Add gradient backgrounds to cards | Visual noise | Use subtle border + shadow depth |
| Use `text-lg` in body | Breaks type scale | Use `text-sm` body, `text-base` titles |
| Add decorative illustrations | Clutters interface | Use the training loss SVG motif, StatusDot, colored chips |
| Create new color variables | Palette drift | Use existing tokens, theme chips for accent shifts |
| Use hardcoded colors | Unmaintainable | Always `rgb(var(--token))` |
| Add animation without `prefers-reduced-motion` check | Accessibility violation | Wrap in `@media (prefers-reduced-motion: no-preference)` |

## Process for New UI Work

1. **Read the existing component** — understand its current structure before changing anything
2. **Identify the visual hierarchy** — what's the one thing the eye should land on?
3. **Check the design system** — does the component use existing tokens and patterns?
4. **Add, don't overhaul** — AGENTS.md rule: "Build Up, Not Overhaul"
5. **Verify** — `npx tsc --noEmit` passes, component renders correctly

## Component Checklist

Before shipping any UI change:
- [ ] All colors via CSS custom properties
- [ ] Spacing uses token system (`gap-*`, `space-y-*`, `p-*`)
- [ ] Typography follows type scale (no `text-lg` in body)
- [ ] Interactive elements have hover/focus/active/disabled states
- [ ] `focus-visible:ring-2 focus-visible:ring-ring` on all interactive elements
- [ ] `aria-label` on icon-only buttons
- [ ] Empty states explain why + provide next action
- [ ] Loading states use skeletons or spinners (never blank)
- [ ] `prefers-reduced-motion` respected for animations
- [ ] `npx tsc --noEmit` passes
