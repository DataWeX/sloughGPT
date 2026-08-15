---
name: component-generation
description: Generate React components that match the existing @sloughgpt/strui component library patterns, TypeScript conventions, and project structure.
---

# Component Generation Skill

## When to Use
- Creating new UI components
- Adding new page sections
- Scaffolding feature components
- Any time a new `.tsx` file is created in `apps/web/`

## Framework Stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 14+ (App Router) |
| Language | TypeScript (strict) |
| Styling | Tailwind CSS v3 + CSS custom properties |
| Component library | `@sloughgpt/strui` (shadcn/ui-based) |
| State | Zustand + React contexts |
| Testing | Vitest + @testing-library/react |
| Icons | `@/components/icons/NavIcons` (re-exports from strui) |

## Component Structure

### Single File Component

```tsx
'use client'

import { cn, Button } from '@sloughgpt/strui'
import { IconSomething } from '@/components/icons/NavIcons'

interface MyComponentProps {
  title: string
  description?: string
  variant?: 'default' | 'secondary'
  onAction?: () => void
}

export function MyComponent({ title, description, variant = 'default', onAction }: MyComponentProps) {
  return (
    <div className="space-y-2">
      <h3 className="text-base font-medium">{title}</h3>
      {description && <p className="text-sm text-muted-foreground">{description}</p>}
      <Button variant={variant} onClick={onAction}>Action</Button>
    </div>
  )
}
```

### Page Component

```tsx
'use client'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

export default function PageName() {
  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Page Title" subtitle="Optional subtitle" />}
      />
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Section Title</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm">Body text here.</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
```

## Import Patterns

```tsx
// UI primitives from strui
import { Button, Card, CardHeader, CardTitle, CardContent, Dialog, Input } from '@sloughgpt/strui'

// Composed components from strui
import { SearchInput, StatCard, EmptyCard, SectionHeader, ListRow } from '@sloughgpt/strui'

// Icons from project re-export
import { IconPlus, IconTrash, IconSettings } from '@/components/icons/NavIcons'

// Project components
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'

// Controllers
import { modelController } from '@/lib/model-controller'
import { sessionController } from '@/lib/session-controller'

// Hooks
import { useLocale } from '@/hooks/useLocale'
import { useSettings } from '@/lib/store'
```

## TypeScript Conventions

```tsx
// Props interface — always explicit, no inline types
interface CardProps {
  title: string
  count: number
  isSelected?: boolean
  onSelect: (id: string) => void
}

// Event handlers — typed explicitly
const handleClick = useCallback((id: string) => {
  onSelect(id)
}, [onSelect])

// State — typed explicitly
const [items, setItems] = useState<Item[]>([])
const [loading, setLoading] = useState(false)

// Refs — typed explicitly
const inputRef = useRef<HTMLInputElement>(null)
```

## Component Patterns

### Card Pattern

```tsx
<Card>
  <CardHeader>
    <CardTitle className="text-base">Title</CardTitle>
  </CardHeader>
  <CardContent className="space-y-3">
    {/* Content */}
  </CardContent>
</Card>
```

### List Pattern (Rich Card)

```tsx
<div className="space-y-2">
  {items.map(item => (
    <div
      key={item.id}
      className={cn(
        'group flex items-center gap-3 rounded-lg border p-3 transition-colors',
        'border-border/60 hover:bg-muted/50',
        selected && 'bg-primary/[0.08] border-primary/40'
      )}
    >
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium truncate">{item.name}</p>
        <p className="text-[10px] text-muted-foreground truncate">{item.description}</p>
      </div>
      <Button size="sm" variant="ghost" className="opacity-0 group-hover:opacity-100">
        Action
      </Button>
    </div>
  ))}
</div>
```

### Empty State Pattern

```tsx
<EmptyCard
  icon={<IconSomething className="h-8 w-8" />}
  title="No items yet"
  description="Create an item to get started"
/>
```

### Loading State Pattern

```tsx
{loading ? (
  <div className="space-y-2">
    <Skeleton className="h-28 rounded-lg" />
    <Skeleton className="h-28 rounded-lg" />
  </div>
) : (
  <div>{/* Content */}</div>
)}
```

### Error State Pattern

```tsx
<ErrorBanner
  message="Failed to load data"
  onRetry={handleRetry}
/>
```

## File Naming

| Type | Pattern | Example |
|------|---------|---------|
| Page | `page.tsx` | `app/(app)/models/page.tsx` |
| Component | `PascalCase.tsx` | `components/ModelCard.tsx` |
| Composed component | `kebab-case.tsx` | `components/training/LossChart.tsx` |
| Hook | `use*.ts` | `hooks/useModelLoader.ts` |
| Controller | `*-controller.ts` | `lib/model-controller.ts` |
| Test | `*.test.tsx` | `components/ModelCard.test.tsx` |
| Context | `*Context.tsx` | `contexts/ModelContext.tsx` |

## Test Pattern

```tsx
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({ usePathname: () => '/test' }))
vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))

import { MyComponent } from './MyComponent'

describe('MyComponent', () => {
  afterEach(cleanup)

  it('renders with title', () => {
    render(<MyComponent title="Test" />)
    expect(screen.getByText('Test')).toBeDefined()
  })

  it('calls onAction when clicked', () => {
    const onAction = vi.fn()
    render(<MyComponent title="Test" onAction={onAction} />)
    screen.getByRole('button').click()
    expect(onAction).toHaveBeenCalled()
  })
})
```

## Anti-Patterns

| Anti-Pattern | Fix |
|-------------|-----|
| Inline styles (`style={{ color: 'red' }}`) | Use Tailwind classes |
| Hardcoded colors (`#7c52c4`) | Use design tokens (`text-primary`) |
| `any` type | Use explicit types |
| `useEffect` for derived state | Use `useMemo` instead |
| Prop drilling > 2 levels | Use context or Zustand |
| Barrel exports in components/ | Import directly from file |
| `console.log` in production | Remove or use dev-only logger |
