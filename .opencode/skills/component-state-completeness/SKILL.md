---
name: component-state-completeness
description: Verify every interactive component has all required states: hover, focus-visible, active/pressed, disabled, loading, success. Flags missing states and ARIA attributes.
---

# Component State Completeness Skill

## When to Use
- Creating new interactive components (buttons, inputs, toggles, links)
- Reviewing PRs that add or modify interactive elements
- Pre-commit audit of UI changes
- Accessibility compliance checks

## Required States for Interactive Elements

### Button States

| State | Visual | Implementation |
|-------|--------|---------------|
| Default | Base appearance | `variant="default"` |
| Hover | Color shift + optional lift | `hover:bg-primary/90` or `hover:-translate-y-0.5 hover:shadow-md` |
| Active/Pressed | Subtle inset | `active:scale-[0.98]` or `active:bg-primary/80` |
| Focus | Ring visible | `focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2` |
| Disabled | Muted, non-interactive | `disabled:opacity-40 disabled:pointer-events-none` |
| Loading | Spinner replaces content | `{loading && <Loader2 className="h-4 w-4 animate-spin" />}` |
| Success | Brief confirmation | `{success && <IconCheck className="h-4 w-4 text-success" />}` |

### Input States

| State | Visual | Implementation |
|-------|--------|---------------|
| Default | Border + background | `border-border bg-card` |
| Hover | Subtle border shift | `hover:border-border/80` |
| Focus | Ring + border color | `focus-visible:ring-2 focus-visible:ring-ring focus-visible:border-primary` |
| Error | Red border + message | `border-destructive` + `aria-describedby="error-id"` |
| Disabled | Muted | `disabled:opacity-40 disabled:cursor-not-allowed` |
| Loading | Spinner indicator | `aria-busy="true"` |

### Toggle/Switch States

| State | Visual | Implementation |
|-------|--------|---------------|
| Off | Gray background | `data-[state=unchecked]:bg-muted` |
| On | Primary color | `data-[state=checked]:bg-primary` |
| Focus | Ring | `focus-visible:ring-2 focus-visible:ring-ring` |
| Disabled | Muted | `disabled:opacity-40` |

## State Checker

```bash
# Find buttons without hover
grep -rn '<button' apps/web/ --include="*.tsx" | grep -v 'hover:\|variant='

# Find buttons without focus-visible
grep -rn '<button' apps/web/ --include="*.tsx" | grep -v 'focus-visible'

# Find inputs without focus styles
grep -rn '<input' apps/web/ --include="*.tsx" | grep -v 'focus-visible\|focus:'

# Find clickable divs without keyboard support
grep -rn 'onClick' apps/web/ --include="*.tsx" | grep -v '<button\|<a\|<Link\|role="button"\|tabIndex'
```

## ARIA Attribute Checklist

| Element | Required Attributes |
|---------|-------------------|
| Icon-only button | `aria-label="Description"` |
| Loading button | `aria-busy="true"` |
| Toggle | `aria-pressed` or `role="switch"` + `aria-checked` |
| Dialog | `role="dialog"` + `aria-modal="true"` + `aria-label` |
| Tab | `role="tab"` + `aria-selected` + `aria-controls` |
| Tab panel | `role="tabpanel"` + `aria-labelledby` |
| Progress | `role="progressbar"` + `aria-valuenow` + `aria-valuemin` + `aria-valuemax` |
| Tooltip | `aria-describedby` pointing to tooltip content |
| Status change | `aria-live="polite"` on container |

## Loading State Pattern

```tsx
function AsyncButton({ onClick, children }: AsyncButtonProps) {
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)

  const handleClick = async () => {
    setLoading(true)
    try {
      await onClick()
      setSuccess(true)
      setTimeout(() => setSuccess(false), 2000)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Button
      onClick={handleClick}
      disabled={loading}
      aria-busy={loading}
    >
      {loading ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : success ? (
        <IconCheck className="h-4 w-4 text-success" />
      ) : (
        children
      )}
    </Button>
  )
}
```

## Icon-Only Button Checklist

Every icon-only button MUST have an accessible label:

```tsx
// BAD - no label
<Button variant="ghost" size="icon">
  <IconTrash />
</Button>

// GOOD - has label
<Button variant="ghost" size="icon" aria-label="Delete item">
  <IconTrash />
</Button>

// GOOD - with tooltip for sighted users
<Tooltip>
  <TooltipTrigger asChild>
    <Button variant="ghost" size="icon" aria-label="Delete item">
      <IconTrash />
    </Button>
  </TooltipTrigger>
  <TooltipContent>Delete item</TooltipContent>
</Tooltip>
```

## Anti-Patterns

| Anti-Pattern | Fix |
|-------------|-----|
| Button with only `onClick` | Add `hover:`, `focus-visible:`, `active:` styles |
| Icon button without `aria-label` | Add descriptive `aria-label` |
| Loading state without `aria-busy` | Add `aria-busy="true"` to loading element |
| Error state without `aria-describedby` | Link error message via `aria-describedby` |
| Toggle without `role="switch"` | Add `role="switch"` + `aria-checked` |
| Dialog without `role="dialog"` | Add `role="dialog"` + `aria-modal="true"` |
| Disabled button still in tab order | Add `tabIndex={-1}` or `aria-disabled` |
