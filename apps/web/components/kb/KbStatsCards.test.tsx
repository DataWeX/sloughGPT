import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { KbStatsCards } from './KbStatsCards'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,

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

describe('KbStatsCards', () => {
  it('renders nothing when stats is null', () => {
    const { container } = render(<KbStatsCards stats={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders all four stat labels', () => {
    render(<KbStatsCards stats={{ total_items: 42, topics: ['a', 'b'], avg_importance: 0.8, source_count: 3 }} />)
    expect(screen.getByText('Total Entries')).toBeDefined()
    expect(screen.getByText('Topics')).toBeDefined()
    expect(screen.getByText('Avg Importance')).toBeDefined()
    expect(screen.getByText('Sources')).toBeDefined()
  })

  it('displays the correct values', () => {
    render(<KbStatsCards stats={{ total_items: 10, topics: ['x'], avg_importance: 0.5, source_count: 2 }} />)
    expect(screen.getByText('10')).toBeDefined()
    expect(screen.getByText('1')).toBeDefined()
    expect(screen.getByText('0.50')).toBeDefined()
    expect(screen.getByText('2')).toBeDefined()
  })

  it('renders four card components', () => {
    render(<KbStatsCards stats={{ total_items: 0, topics: [], avg_importance: 0, source_count: 0 }} />)
    expect(screen.getAllByTestId('card')).toHaveLength(4)
  })

  it('formats importance to two decimal places', () => {
    render(<KbStatsCards stats={{ total_items: 1, topics: [], avg_importance: 0.123, source_count: 0 }} />)
    expect(screen.getByText('0.12')).toBeDefined()
  })
})
