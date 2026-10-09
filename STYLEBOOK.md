# sloughGPT Design System — Noir Violet

Unified design language for web (Next.js + Tailwind) and mobile (React Native + Tamagui).

---

## Palette — Noir Violet

All colors use RGB triples. CSS uses `rgb(var(--token))`, Tamagui uses hex literals.

### Core

| Role | Light | Dark | CSS Variable |
|------|-------|------|-------------|
| Primary | `124 82 196` (#7C52C4) | `192 170 244` (#C0AAF4) | `--primary` |
| Accent | `236 145 95` (#EC915F) | `240 176 130` (#F0B082) | `--accent` |
| Background | `248 246 252` (#F8F6FC) | `17 15 24` (#110F18) | `--background` |
| Card | `255 255 255` (#FFFFFF) | `28 25 38` (#1C1926) | `--card` |
| Border | `228 224 242` (#E4E0F2) | `52 46 72` (#342E48) | `--border` |
| Chat BG | `246 242 237` (#F6F2ED) | `22 20 30` (#16141E) | `--chat-bg` |

### Text

| Role | Light | Dark |
|------|-------|------|
| Foreground | `25 22 36` (#191624) | `238 234 248` (#EEEAF8) |
| Muted | `130 122 150` (#827A96) | `150 140 172` (#968CAC) |
| Secondary | `42 37 55` (#2A2537) | `238 234 248` (#EEEAF8) |
| On Primary | `250 248 255` (#FAF8FF) | `25 22 36` (#191624) |

### Semantic

| Role | Light | Dark |
|------|-------|------|
| Success | `52 176 125` (#34B07D) | `72 192 140` (#48C08C) |
| Warning | `236 168 60` (#ECA83C) | `240 192 80` (#F0C050) |
| Error | `220 80 90` (#DC505A) | `235 100 110` (#EB646E) |
| Info | `90 150 220` (#5A96DC) | `120 175 240` (#78AFF0) |

### Elevation

| Token | Light | Dark |
|-------|-------|------|
| Shadow SM | `0 1px 2px rgba(25,22,36,0.06)` | `0 1px 2px rgba(0,0,0,0.25)` |
| Shadow MD | `0 4px 12px rgba(25,22,36,0.08)` | `0 4px 12px rgba(0,0,0,0.35)` |
| Shadow LG | `0 8px 30px rgba(25,22,36,0.10)` | `0 8px 30px rgba(0,0,0,0.45)` |

---

## Typography

### Font Stack

| Role | Web | Mobile |
|------|-----|--------|
| Body | Rubik (400/500/600) | Outfit (400/500/600/700) |
| Mono | JetBrains Mono (400/500) | JetBrains Mono (400/500) |
| Numeric | Lato | System |

### Type Scale

| Role | Web Class | Mobile Value | Usage |
|------|-----------|-------------|-------|
| Page title | `sl-h1` text-2xl/3xl | 20-28px, 700 | One per page |
| Section title | `text-base font-medium` | 14-15px, 600 | Card headers |
| Body | `text-sm leading-relaxed` | 13-14px, 400 | Primary content |
| Caption | `text-xs text-muted-foreground` | 10-12px, 400 | Timestamps, meta |
| Badge | `text-[10px] font-medium` | 10-11px, 500 | Status badges, tags |
| Code | `text-xs font-mono` | 12px mono | Model names, tokens |

### Rules

- Never use `text-lg` or `text-2xl` in body content
- `tracking-tight` on titles, `tracking-wider uppercase` on section dividers
- `leading-relaxed` on message text for readability
- Touch targets minimum 44px (h-11)

---

## Spacing

### Tokens

| Name | Value | Usage |
|------|-------|-------|
| xs | 4px | Inline gaps, icon padding |
| sm | 8px | Tight spacing, badge padding |
| md | 12px | Card internal gaps |
| lg | 16px | Page padding, section gaps |
| xl | 20px | Large section gaps |
| 2xl | 24px | Page margin, hero spacing |

### Layout Constants

| Token | Web | Mobile |
|-------|-----|--------|
| Page padding | 12-32px responsive | 16px |
| Card padding | 20px (p-5) | 12-16px |
| Section gap | 16px (space-y-4) | 12-16px |
| Card radius | 8px (rounded-lg) | 10-12px |
| Button radius | 8px (rounded-lg) | 8-12px |
| Bubble radius | 16px (rounded-2xl) | 16px |
| Input radius | 16px (rounded-2xl) | 16px |
| Badge radius | 999px (rounded-full) | 999px |
| Sidebar width | 208px | Drawer |
| Chat thread max | 832px | Full width |

---

## Components

### Cards

**Web:** `rounded-lg border bg-card text-card-foreground shadow-sm`
**Mobile:** `bg-card border rounded-xl p-4`

Variants:
- `default` — white/dark card with border
- `interactive` — hover lift (web), press feedback (mobile)
- `selected` — primary border/bg tint
- `error` — destructive border

### Buttons

**Web:** `rounded-lg text-sm font-medium transition-all duration-200 active:scale-[0.98]`
**Mobile:** `bg-primary text-white rounded-lg font-medium h-11`

| Variant | Web Style | Mobile Style |
|---------|-----------|-------------|
| Primary | `bg-primary text-primary-foreground` | `bg-primary text-white` |
| Secondary | `border bg-secondary text-secondary-foreground` | `border bg-secondary text-foreground` |
| Ghost | `text-muted-foreground hover:bg-accent/8` | `bg-transparent text-muted` |
| Destructive | `bg-destructive text-destructive-foreground` | `bg-error text-white` |
| Outline | `border bg-transparent text-foreground` | `border bg-transparent text-foreground` |

Sizes: h-9 (sm), h-10 (md), h-11 (lg), h-12 (xl)

### Inputs

**Web:** `flex w-full rounded-lg border border-border bg-background px-3 py-2 text-sm shadow-sm focus-visible:ring-2 focus-visible:ring-primary/30`
**Mobile:** `border border-border rounded-xl px-4 py-3 text-sm bg-background`

Height: 44px minimum (h-11)

### Badges

**Web:** `inline-flex items-center rounded-full border font-medium`
**Mobile:** `rounded-full px-2 py-0.5 text-xs font-medium`

| Variant | Web | Mobile |
|---------|-----|--------|
| Default | Secondary bg | Secondary bg |
| Primary | Primary/15 bg | Primary/15 bg |
| Success | Success/15 bg | Success/15 bg |
| Warning | Warning/15 bg | Warning/15 bg |
| Error | Destructive/15 bg | Error/15 bg |

Sizes: sm (10px), default (11-12px), lg (13px)

### Message Bubbles

**User:** `bg-primary text-primary-foreground rounded-2xl rounded-br-md shadow-md`
**Assistant:** `bg-card text-foreground rounded-2xl rounded-bl-md border border-border/40 shadow-sm`

Width: max-w-[90%] mobile, max-w-[72%] desktop
Role labels: 10px, uppercase, tracking-wider

### Chat Input Bar

**Web:** `flex items-end gap-2 rounded-2xl border border-border/50 bg-card px-3 py-2 shadow-sm focus-within:border-primary/40 focus-within:shadow-md`
**Mobile:** Same pattern — rounded-2xl, border, card bg, focus ring

Send button: 44px circle, primary bg, white icon

### Toast

**Web:** `rounded-[10px] border backdrop-blur(20px)` with 3px left accent stripe
**Mobile:** `rounded-xl border backdrop-blur` with left accent stripe

Animation: slide-in + scale (0.3s), fade-out (0.2s)

---

## Chat UI Patterns

### Empty State

- MoodOrb (animated gradient sphere) or AI icon
- Greeting text (time-aware)
- 4 suggestion chips as cards with icons

### Message List

- User messages: right-aligned, solid primary bubble
- Assistant messages: left-aligned, card bubble with border
- Date dividers: thin line + centered date text
- Thinking indicator: "Thinking..." with reasoning panel
- Streaming cursor: pulsing block

### Input Area

- Auto-resizing textarea (max 160px)
- Left accessories: attachment, slash commands, quick prompts
- Right: send button (44px circle)
- Keyboard-aware: scrolls up when keyboard opens

### Sidebar (Mobile)

- Drawer overlay with backdrop blur
- Session list with recency grouping
- New chat button at top
- Swipe to delete

---

## Animations

### Web

- `transition-all duration-200` on interactive elements
- `hover:-translate-y-0.5 hover:shadow-md` on clickable cards
- `active:scale-[0.98]` on buttons
- `animate-in fade-in` on page mount
- `prefers-reduced-motion` respected

### Mobile

- All animations use React Native `Animated` API
- `useNativeDriver: true` for performance
- Common durations: 200-300ms
- Easing: `Easing.out(Easing.cubic)` for entrances
- `prefers-reduced-motion` respected via `AccessibilityInfo`

---

## Anti-Patterns

| Don't | Do Instead |
|-------|-----------|
| Hardcode colors in components | Use `useColors()` hook or theme tokens |
| Use arbitrary spacing values | Use spacing tokens (xs/sm/md/lg/xl/2xl) |
| Use `text-lg` in body | Use body scale (13-14px) |
| Skip loading states | Show skeleton or spinner |
| Use flat colors without depth | Use elevation tokens for shadows |
| Break touch target minimum | Minimum 44px height |
| Skip accessibility labels | Add `aria-label` / `accessibilityLabel` |

---

## File Locations

| File | Purpose |
|------|---------|
| `apps/web/app/globals.css` | Web CSS variables, all tokens |
| `apps/mobile/tamagui.config.ts` | Mobile theme definitions |
| `apps/mobile/src/theme/colors.ts` | Mobile runtime color access |
| `apps/mobile/src/theme/TamaguiProvider.tsx` | Mobile theme switching |
| `STYLEBOOK.md` | This file — unified reference |
