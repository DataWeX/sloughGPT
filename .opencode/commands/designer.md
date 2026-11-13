---
description: >
  Run UI/UX design review and implementation. Usage: /designer <page_or_component>
  Audits and improves the specified page or component against the Noir Violet design system.
agent: designer
---

# Designer Command

Review and improve UI/UX for the Noir Violet design system.

## Usage

```
/designer <page_or_component>
```

## Examples

```
/designer apps/web/app/(app)/chat/page.tsx
/designer apps/web/features/chat/components/ChatInput.tsx
/designer apps/web/components/models/ModelCard.tsx
```

## What It Does

1. **Audit** — Check against Noir Violet design system rules
2. **Identify** — Find violations (hardcoded colors, missing states, bad typography)
3. **Fix** — Apply targeted corrections
4. **Verify** — Ensure no regressions in adjacent components

## Design System Rules

| Rule | Check |
|------|-------|
| Colors | `rgb(var(--token))` only, no hex/hsl |
| Typography | Outfit (sans) + JetBrains Mono (mono) |
| Spacing | `sl-page mx-auto max-w-4xl`, `space-y-4` |
| Components | `@sloughgpt/strui` primitives |
| States | hover, focus-visible, active, disabled, loading |
| Accessibility | `aria-label`, `role`, keyboard nav |
| Responsive | Mobile-first breakpoints |

## Verification

```bash
cd apps/web && npx tsc --noEmit
npm run test:components
```

## Output

The agent will report:
1. Violations found
2. Changes made
3. Verification results
