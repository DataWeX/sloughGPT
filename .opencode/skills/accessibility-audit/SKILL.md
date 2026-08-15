---
name: accessibility-audit
description: WCAG 2.2 AA compliance audit for React/Next.js components. Checks ARIA, keyboard navigation, contrast, target sizes, focus management, and screen reader support.
---

# Accessibility Audit Skill

## When to Use
- Before committing any UI component changes
- When creating new components or pages
- When reviewing PRs that touch frontend code
- When a11y issues are reported

## WCAG 2.2 AA Checklist

### 1. Color Contrast (Minimum 4.5:1 text, 3:1 UI)

```bash
# Check for hardcoded colors that bypass the token system
grep -rn '#[0-9a-fA-F]\{3,8\}' apps/web/components/ --include="*.tsx" | grep -v 'test\|stories\|\.test\.'
grep -rn 'rgb(' apps/web/components/ --include="*.tsx" | grep -v 'var(--'
```

**Fix:** Replace with design tokens: `text-foreground`, `bg-primary`, `rgb(var(--token))`

### 2. Keyboard Navigation

Every interactive element must be reachable via Tab/Enter/Escape.

| Requirement | Implementation |
|-------------|---------------|
| Focus visible | `focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2` |
| Tab order | Logical DOM order, no `tabIndex > 0` |
| Enter activates | `<button>`, `<a>`, `<Link>` |
| Escape closes | Dialogs, modals, popovers |
| Arrow keys | Dropdowns, tabs, menus |

**Check:**
```bash
# Find interactive elements without focus styles
grep -rn 'onClick\|href' apps/web/components/ --include="*.tsx" | grep -v 'focus-visible\|focus:\|onFocus\|onBlur'
```

### 3. ARIA Attributes

| Element | Required ARIA |
|---------|--------------|
| Icon-only button | `aria-label="Description"` |
| Dialog/Modal | `role="dialog" aria-modal="true" aria-label="Title"` |
| Tab list | `role="tablist"` on container, `role="tab"` + `aria-selected` on tabs |
| Tab panel | `role="tabpanel"` with `aria-labelledby` |
| Status change | `aria-live="polite"` on container |
| Hidden decorative | `aria-hidden="true"` |
| Loading state | `aria-busy="true"` |

**Check:**
```bash
# Find icon-only buttons missing aria-label
grep -rn '<button' apps/web/components/ --include="*.tsx" | grep -v 'aria-label' | grep -v 'children\|>\s*[A-Z]'

# Find dialogs missing role
grep -rn 'Dialog\|Modal' apps/web/components/ --include="*.tsx" | grep -v 'role='
```

### 4. Touch Target Size (WCAG 2.2)

| Target | Minimum Size |
|--------|-------------|
| Interactive elements | 24x24px minimum |
| Touch targets (mobile) | 44x44px (Apple HIG) / 48x48px (Material) |
| Buttons | `h-7` (28px) inline, `h-9` (36px) default, `h-11` (44px) primary CTA |
| Icon-only buttons | `h-7 w-7` minimum with `aria-label` |

**Check:**
```bash
# Find small clickable elements
grep -rn 'h-5 w-5\|h-4 w-4' apps/web/components/ --include="*.tsx" | grep -v 'aria-label\|title='
```

### 5. Screen Reader Support

| Pattern | Implementation |
|---------|---------------|
| Decorative images | `alt=""` or `aria-hidden="true"` |
| Status updates | `aria-live="polite"` with `aria-atomic="true"` |
| Hidden content | `className="sr-only"` (not `display: none`) |
| Page structure | Semantic HTML: `<nav>`, `<main>`, `<header>`, `<aside>`, `<footer>` |
| Lists | `<ul>`/`<ol>` for list content, not `<div>` with bullets |

### 6. Reduced Motion

```tsx
// Respect user preference
<div className="motion-safe:animate-bounce motion-reduce:animate-none" />

// Or in CSS
@media (prefers-reduced-motion: reduce) {
  * { animation-duration: 0.01ms !important; }
}
```

### 7. Form Accessibility

| Requirement | Implementation |
|-------------|---------------|
| Labels | Every `<input>` has `<label>` or `aria-label` |
| Error messages | `aria-describedby` pointing to error text |
| Required fields | `aria-required="true"` or `required` attribute |
| Fieldsets | Group related fields with `<fieldset>` + `<legend>` |

## Verification Steps

1. **Keyboard test:** Tab through all interactive elements on the page. Every element must show a visible focus ring.
2. **Screen reader test:** Navigate with VoiceOver (Mac) or NVDA (Windows). All content must be announced logically.
3. **Zoom test:** Zoom to 200%. No content should be cut off or overlap.
4. **Color test:** Check contrast ratios with browser dev tools or axe extension.
5. **Automated scan:** Run axe-core or lighthouse accessibility audit.

## Anti-Patterns

| Anti-Pattern | Fix |
|-------------|-----|
| `<div onClick>` without keyboard support | Use `<button>` or add `role="button" tabIndex={0} onKeyDown` |
| Missing `aria-label` on icon buttons | Add descriptive label: `aria-label="Close dialog"` |
| `tabIndex={100}` | Remove —破坏 natural tab order |
| `outline: none` without replacement | Use `focus-visible:ring-2` instead |
| Color as only indicator | Add icon/text alongside color |
| Auto-playing video/audio | Provide controls and pause mechanism |
| `display: none` for screen reader hiding | Use `sr-only` class instead |
