---
name: strui-component-integration
description: How to import, use, and test @sloughgpt/strui components correctly. Covers import paths, subpath barrels, mock patterns, and Storybook reference.
---

# Strui Component Integration Skill

## When to Use
- Importing strui components in `apps/web/`
- Writing tests that mock strui
- Creating components that compose strui primitives
- Debugging strui-related build or test issues

## Import Paths

### Direct Imports (Preferred)

```tsx
// UI primitives
import { Button, Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@sloughgpt/strui'
import { Input, Label, Textarea } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { Tooltip, TooltipTrigger, TooltipContent } from '@sloughgpt/strui'

// Composed components
import { SearchInput, StatCard, EmptyCard, SectionHeader } from '@sloughgpt/strui'
import { ListRow, KpiGrid, Breadcrumbs } from '@sloughgpt/strui'

// AI-specific components
import { ModelPicker, PromptComposer, MessageBubble } from '@sloughgpt/strui'

// Hooks
import { useToast } from '@sloughgpt/strui'

// Utilities
import { cn, buttonVariants } from '@sloughgpt/strui'
```

### Subpath Imports (If Needed)

```tsx
import { Button } from '@sloughgpt/strui/ui'
import { StatCard } from '@sloughgpt/strui/composed'
import { ModelPicker } from '@sloughgpt/strui/ai'
```

## Component Catalog

### UI Primitives (26)

| Component | Import | Key Props |
|-----------|--------|-----------|
| Alert | `Alert, AlertTitle, AlertDescription` | `variant: 'default' \| 'destructive'` |
| AlertDialog | `AlertDialog, AlertDialogTrigger, AlertDialogContent` | `open`, `onOpenChange` |
| Avatar | `Avatar, AvatarImage, AvatarFallback` | `src`, `fallback` |
| Badge | `Badge` | `variant: 'default' \| 'secondary' \| 'destructive'` |
| Button | `Button` | `variant: 'default' \| 'ghost' \| 'menu' \| 'link'`, `size: 'default' \| 'sm' \| 'icon'` |
| Card | `Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter` | — |
| Checkbox | `Checkbox` | `checked`, `onCheckedChange` |
| Collapsible | `Collapsible, CollapsibleTrigger, CollapsibleContent` | `open`, `onOpenChange` |
| Dialog | `Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription` | `open`, `onOpenChange` |
| DropdownMenu | `DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem` | `open`, `onOpenChange` |
| Icons | `IconPlus, IconTrash, IconSettings, ...` (50+) | `className` |
| Input | `Input` | `type`, `placeholder`, `value`, `onChange` |
| Label | `Label` | `htmlFor` |
| Popover | `Popover, PopoverTrigger, PopoverContent` | `open`, `onOpenChange` |
| Progress | `Progress` | `value` (0-100) |
| Select | `Select, SelectTrigger, SelectContent, SelectItem` | `value`, `onValueChange` |
| Separator | `Separator` | `orientation: 'horizontal' \| 'vertical'` |
| Slider | `Slider` | `value`, `onValueChange`, `min`, `max` |
| Switch | `Switch` | `checked`, `onCheckedChange` |
| Tabs | `Tabs, TabsList, TabsTrigger, TabsContent` | `value`, `onValueChange` |
| Textarea | `Textarea` | `value`, `onChange` |
| Toast | `toast, ToastProvider` | `variant: 'default' \| 'destructive'` |
| ToggleGroup | `ToggleGroup, ToggleGroupItem` | `value`, `onValueChange`, `type: 'single' \| 'multiple'` |
| Tooltip | `Tooltip, TooltipTrigger, TooltipContent` | `delayDuration` |

### Composed Components (22)

| Component | Import | Purpose |
|-----------|--------|---------|
| AppShell | `AppShell` | Page layout wrapper |
| Breadcrumbs | `Breadcrumbs` | Navigation breadcrumbs |
| Chip | `Chip` | Tag/chip with optional close |
| CopyButton | `CopyButton` | Copy to clipboard button |
| EmptyCard | `EmptyCard` | Empty state placeholder |
| EmptyState | `EmptyState` | Full-page empty state |
| FormField | `FormField` | Label + Input + Error wrapper |
| InlineBanner | `InlineBanner` | In-page notification |
| Kbd | `Kbd` | Keyboard shortcut display |
| KeyValueList | `KeyValueList` | Key-value pairs list |
| KpiGrid | `KpiGrid` | KPI metric grid |
| ListRow | `ListRow` | Rich list item |
| NavRail | `NavRail` | Side navigation |
| PageHeader | `PageHeader` | Page header with actions |
| ProgressBar | `ProgressBar` | Progress indicator |
| ScrollPanel | `ScrollPanel` | Scrollable container |
| SearchInput | `SearchInput` | Search input with icon |
| SectionHeader | `SectionHeader` | Section title with actions |
| SettingsRow | `SettingsRow` | Settings toggle row |
| Skeleton | `Skeleton` | Loading skeleton |
| StatCard | `StatCard` | KPI stat card |
| StatusDot | `StatusDot` | Status indicator dot |

### AI Components (16)

| Component | Import | Purpose |
|-----------|--------|---------|
| ChatLayout | `ChatLayout` | Chat page layout |
| ChatThread | `ChatThread` | Message list |
| MessageBubble | `MessageBubble` | Chat message |
| PromptComposer | `PromptComposer` | Message input |
| ModelPicker | `ModelPicker` | Model selection |
| CodeSnippet | `CodeSnippet` | Code block |
| ToolCallCard | `ToolCallCard` | Tool call display |
| ReasoningPanel | `ReasoningPanel` | Chain-of-thought |
| TokenMeter | `TokenMeter` | Token usage display |
| TypingIndicator | `TypingIndicator` | Loading dots |

## Test Mock Patterns

### Mocking the Entire Module

```tsx
vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <h3 {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  Input: (props: any) => <input {...props} />,
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
  Dialog: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  DialogContent: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  useToast: () => ({ toast: vi.fn() }),
}))
```

### Mocking Icons

```tsx
vi.mock('@/components/icons/NavIcons', () => {
  const stub = (name: string) => {
    const Component = (props: any) => <svg data-testid={`icon-${name}`} {...props} />
    Component.displayName = name
    return Component
  }
  return {
    IconPlus: stub('IconPlus'),
    IconTrash: stub('IconTrash'),
    IconSettings: stub('IconSettings'),
    IconClose: stub('IconClose'),
  }
})
```

### Mocking next/navigation

```tsx
vi.mock('next/navigation', () => ({
  usePathname: () => '/models',
  useRouter: () => ({ push: vi.fn(), back: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}))
```

### Mocking useLocale

```tsx
vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({
    t: (key: string, params?: Record<string, any>) => {
      if (params) {
        return Object.entries(params).reduce(
          (str, [k, v]) => str.replace(new RegExp(`\\{${k}\\}`, 'g'), String(v)),
          key
        )
      }
      return key
    },
    locale: 'en',
  }),
  LOCALES: ['en', 'es', 'fr', 'de', 'zh'],
}))
```

## Anti-Patterns

| Anti-Pattern | Fix |
|-------------|-----|
| `import { Button } from '@/components/ui/button'` | Use `import { Button } from '@sloughgpt/strui'` |
| `import * as strui from '@sloughgpt/strui'` | Import specific components |
| Mocking entire strui in every test | Only mock what you need, prefer real components |
| Using `any` in mock types | Use proper component props |
| Not cleaning up after render | Add `afterEach(cleanup)` |
| Missing `data-testid` in mocks | Add test IDs for assertion |
