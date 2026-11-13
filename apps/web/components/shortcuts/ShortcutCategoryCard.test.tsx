/// <reference types="vitest" />
import { render, screen, cleanup, within } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { ShortcutCategoryCard, type Shortcut } from './ShortcutCategoryCard'
import { Keyboard, RefreshCw } from 'lucide-react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}))

afterEach(() => cleanup())

const shortcuts: Shortcut[] = [
  { keys: ['Ctrl', 'K'], label: 'Command palette', icon: Keyboard },
  { keys: ['R'], label: 'Refresh data', icon: RefreshCw },
]

describe('ShortcutCategoryCard', () => {
  it('renders the category title', () => {
    render(<ShortcutCategoryCard title="Global" shortcuts={[]} />)
    expect(screen.getByText('Global')).toBeDefined()
  })

  it('renders shortcut labels', () => {
    render(<ShortcutCategoryCard title="Global" shortcuts={shortcuts} />)
    expect(screen.getByText('Command palette')).toBeDefined()
    expect(screen.getByText('Refresh data')).toBeDefined()
  })

  it('renders kbd keys for each shortcut', () => {
    render(<ShortcutCategoryCard title="Global" shortcuts={shortcuts} />)
    const kbds = screen.getAllByText('Ctrl')
    expect(kbds.length).toBeGreaterThanOrEqual(1)
  })

  it('renders with empty shortcuts array', () => {
    render(<ShortcutCategoryCard title="Empty" shortcuts={[]} />)
    expect(screen.getByText('Empty')).toBeDefined()
    const content = screen.getByTestId('card-content')
    expect(within(content).queryByText('+')).toBeNull()
  })

  it('renders plus separator between multi-key combos', () => {
    render(<ShortcutCategoryCard title="Test" shortcuts={shortcuts} />)
    const separators = screen.getAllByText('+')
    expect(separators.length).toBeGreaterThanOrEqual(1)
  })
})
