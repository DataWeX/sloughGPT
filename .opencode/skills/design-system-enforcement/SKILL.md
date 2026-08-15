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

| Token | Light | Dark | Tailwind |
|-------|-------|------|----------|
| `--primary` | `124 82 196` | `192 170 244` | `text-primary`, `bg-primary` |
| `--accent` | `236 145 95` | `240 176 130` | `text-accent`, `bg-accent` |
| `--success` | `52 176 125` | `72 192 140` | `text-success`, `bg-success` |
| `--warning` | `236 168 60` | `240 192 80` | `text-warning`, `bg-warning` |
| `--destructive` | `220 80 90` | `235 100 110` | `text-destructive`, `bg-destructive` |
| `--background` | `248 246 252` | `17 15 24` | `bg-background` |
| `--card` | `255 255 255` | `28 25 38` | `bg-card` |
| `--border` | `228 224 242` | `52 46 72` | `border-border` |
| `--muted` | `244 242 248` | `38 34 52` | `bg-muted` |
| `--muted-foreground` | `130 122 150` | `150 140 172` | `text-muted-foreground` |

### Typography Scale

| Role | Class | Size | Weight |
|------|-------|------|--------|
| Page title | `text-2xl md:text-3xl` | 600 | Only in `AppRouteHeaderLead` |
| Section title | `text-base` | 500 | Card headers |
| Body | `text-sm` | 400 | Primary content |
| Caption | `text-xs text-muted-foreground` | 400 | Secondary info |
| Label | `text-xs font-medium uppercase tracking-wider` | 500 | Form labels |
| Badge | `text-[10px] font-medium` | 500 | Status badges |

### Spacing

| Token | Value | Usage |
|-------|-------|-------|
| `sl-page` | `p-4 sm:p-6 md:p-8` | Page wrapper |
| `max-w-4xl` | 896px | Max content width |
| `space-y-4` | 16px | Between cards |
| `gap-1` through `gap-4` | 4-16px | Component gaps |

## Violation Scanner

### Hardcoded Colors

```bash
# Find hex colors in components
grep -rn '#[0-9a-fA-F]\{3,8\}' apps/web/ --include="*.tsx" --include="*.css" | grep -v 'node_modules\|\.test\.\|stories\.'

# Find rgb()/rgba() not using var()
grep -rn 'rgb(' apps/web/ --include="*.tsx" --include="*.css" | grep -v 'var(--'
```

**Fix:** Replace with token: `text-primary`, `bg-border`, `rgb(var(--primary))`

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

## Component Checklist

Before committing any UI component, verify:

| Check | Pass |
|-------|------|
| Colors use tokens (no hex, no rgb without var) | |
| Typography follows scale (no text-lg in body) | |
| Spacing uses system (no arbitrary px values) | |
| Interactive elements have hover states | |
| Focus states use ring-2 + ring-offset-2 | |
| Disabled states use opacity-40 | |
| Cards use CardHeader/CardTitle/CardContent | |
| Page uses `sl-page mx-auto max-w-4xl` | |
| No inline styles | |
| No console.log in production | |
| Component has TypeScript interface for props | |
| Component has test file | |

## Anti-Patterns

| Anti-Pattern | Fix |
|-------------|-----|
| `text-gray-500` | `text-muted-foreground` |
| `bg-white` | `bg-card` |
| `border-gray-200` | `border-border` |
| `text-lg` in page body | `text-sm` or `text-base` |
| `px-8 py-6` on page | `sl-page` class |
| `rounded-lg` everywhere | Use appropriate radius for context |
| No hover on button | Add `hover:bg-primary/90` |
| Focus without ring | Add `focus-visible:ring-2 focus-visible:ring-ring` |
