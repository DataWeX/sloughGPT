# Noir Violet Design System — LOCKED

This is the **only** design system for sloughGPT. It does not change. All UI work must follow these rules exactly. Do not invent new colors, fonts, spacing, or patterns.

## Identity

**Name:** Noir Violet  
**Character:** Warm, sophisticated, technical. Rich violet primary with warm terracotta accents.  
**Mood:** Calm confidence. Not sterile, not playful. A tool that respects its user.

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

## Typography

### Font Families

| Role | Font | Fallback |
|------|------|----------|
| Body | Rubik (`--font-rubik`) | system-ui, sans-serif |
| Numeric | Lato (`--font-lato`) | system-ui, sans-serif |
| Code | JetBrains Mono (`--font-jetbrains-mono`) | ui-monospace, monospace |

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
| New font families | Use Rubik, Lato, or JetBrains Mono |
| `rounded-full` on cards | Use `rounded` or `rounded-lg` |
| Animations without motion check | Add `motion-reduce:` variants |

## AI Agent Rules

1. **This design system is locked.** Do not propose changes to colors, fonts, spacing, or component patterns.
2. **Do not use the frontend-design skill** for sloughGPT. It is for new projects only.
3. **Always use existing tokens.** Never invent new colors or spacing values.
4. **Always use strui components.** Do not build custom UI primitives.
5. **Always follow the type scale.** No arbitrary font sizes.
6. **Always add hover states.** No clickable elements without feedback.
7. **Always use focus rings.** Accessibility is not optional.
8. **Test in both light and dark mode.** Every component must work in both.
