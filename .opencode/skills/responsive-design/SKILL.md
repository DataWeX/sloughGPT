---
name: responsive-design
description: Mobile-first responsive design patterns for Next.js + Tailwind CSS. Enforces breakpoint usage, touch targets, fluid layouts, and mobile-specific patterns.
---

# Responsive Design Skill

## When to Use
- Creating new page layouts or components
- Fixing mobile rendering issues
- Adding responsive behavior to existing components
- Reviewing responsive-related PRs

## Breakpoint System

| Prefix | Min-width | Usage |
|--------|-----------|-------|
| (none) | 0px | Mobile-first default |
| `sm` | 640px | Small tablets |
| `md` | 768px | Tablets |
| `lg` | 1024px | Desktop |
| `xl` | 1280px | Large desktop |

**Rule:** Always start mobile-first. Add complexity with `sm:`, `md:`, `lg:` prefixes.

## Page Layout Pattern

```tsx
<div className="sl-page mx-auto max-w-4xl">
  <AppRouteHeader left={<AppRouteHeaderLead title="..." subtitle="..." />} />
  <div className="space-y-4">
    <Card> ... </Card>
  </div>
</div>
```

| Token | Value | Usage |
|-------|-------|-------|
| `sl-page` | `p-4 sm:p-6 md:p-8` | Responsive page padding |
| `max-w-4xl` | 896px | Max content width |
| `space-y-4` | 16px | Gap between cards |

## Mobile-First Patterns

### Sidebar/Navigation

| Pattern | Mobile | Desktop |
|---------|--------|---------|
| Navigation | Drawer (hamburger menu) | Persistent sidebar |
| Sidebar width | Full screen width | Collapsible panel |
| Close mechanism | X button + backdrop | Toggle button |
| Safe area | `pb-[max(0px,env(safe-area-inset-bottom))]` | N/A |

```tsx
// Mobile drawer
{portalMounted && createPortal(
  <div className={cn('sl-app-drawer-backdrop', open ? 'opacity-100' : 'opacity-0 pointer-events-none')} />
  <div className={cn('sl-app-drawer', open ? 'translate-x-0' : '-translate-x-full pointer-events-none')}>
    <Sidebar variant="drawer" onClose={close} />
  </div>,
  document.body
)}
```

### Message Bubbles (Chat)

```tsx
// Responsive max-width: wider on mobile, constrained on desktop
<div className={cn(
  'max-w-[85%] sm:max-w-[70%] lg:max-w-[60%]',
  'rounded-xl px-4 py-2'
)}>
```

### Touch Targets

| Element | Mobile Size | Desktop Size |
|---------|-------------|-------------|
| Buttons | `h-11` (44px) | `h-9` (36px) |
| Icon buttons | `h-11 w-11` | `h-7 w-7` |
| List items | `min-h-11` | `min-h-9` |
| Input fields | `h-11` | `h-9` |

### Header/Toolbar

```tsx
// Hide complex elements on mobile, show simplified version
<header className="flex items-center gap-2 px-3 h-12">
  {/* Always visible */}
  <Button variant="menu" size="icon" className="lg:hidden">
    <IconMenu />
  </Button>
  
  {/* Mobile: simplified */}
  <span className="text-sm font-semibold truncate lg:hidden">Title</span>
  
  {/* Desktop: full toolbar */}
  <div className="hidden lg:flex lg:items-center lg:gap-4">
    <SearchInput />
    <ModelDropdown />
    <SettingsToggle />
  </div>
</header>
```

## Responsive Utilities

### Text Truncation

```tsx
// Single line
<span className="truncate">Long text</span>

// Multi line
<span className="line-clamp-2">Long text that wraps to two lines max</span>
```

### Visibility

```tsx
// Hide on mobile
<div className="hidden sm:block">Desktop only</div>

// Hide on desktop
<div className="sm:hidden">Mobile only</div>

// Show at specific breakpoint
<div className="hidden md:block lg:hidden">Tablet only</div>
```

### Flex/Grid Responsiveness

```tsx
// Stack on mobile, row on desktop
<div className="flex flex-col sm:flex-row gap-4">

// Grid: 1 col mobile, 2 col tablet, 3 col desktop
<div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
```

## Container Queries (Emerging)

```tsx
// Component-level responsiveness
<div className="@container">
  <div className="flex flex-col @md:flex-row @md:items-center">
    {/* Adapts to parent width, not viewport */}
  </div>
</div>
```

## Anti-Patterns

| Anti-Pattern | Fix |
|-------------|-----|
| `px-8 py-6` on page body | Use `sl-page` class instead |
| Fixed widths `w-[500px]` | Use `max-w-*` with responsive prefixes |
| `overflow-x: auto` on body | Fix layout to not overflow |
| Mobile layout breaking on zoom | Use relative units, test at 200% zoom |
| Touch targets < 44px | Use `min-h-11` or `h-11` on interactive elements |
| Hidden content still focusable | Remove from tab order when hidden |
| No `env(safe-area-inset-*)` on mobile | Add safe area padding for notch devices |

## Verification

1. **Resize test:** Drag browser from 320px to 1440px width. No horizontal scroll at any width.
2. **Device test:** Test on actual phone (iOS Safari, Chrome Android). Touch targets feel right.
3. **Orientation test:** Rotate phone. Layout adapts without breaking.
4. **Zoom test:** Browser zoom to 200%. Content readable, no overflow.
5. **Keyboard test:** Tab through mobile layout. All elements reachable.
