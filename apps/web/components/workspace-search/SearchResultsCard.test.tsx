/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { SearchResultsCard } from './SearchResultsCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
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

const mockGroups = [
  {
    type: 'member',
    label: 'Members',
    color: 'bg-purple-100 text-purple-700',
    link: '/workspaces',
    items: [
      { id: '1', type: 'member', title: 'alice', detail: 'alice@test.com' },
    ],
  },
  {
    type: 'dataset',
    label: 'Datasets',
    color: 'bg-green-100 text-green-700',
    link: '/datasets',
    items: [
      { id: '2', type: 'dataset', title: 'wiki-text', detail: 'Wikipedia dump' },
    ],
  },
]

describe('SearchResultsCard', () => {
  it('renders result count', () => {
    render(<SearchResultsCard groups={mockGroups} total={2} query="test" />)
    expect(screen.getByText('2 result(s) found')).toBeDefined()
  })

  it('renders group labels', () => {
    render(<SearchResultsCard groups={mockGroups} total={2} query="test" />)
    expect(screen.getByText('Members')).toBeDefined()
    expect(screen.getByText('Datasets')).toBeDefined()
  })

  it('renders item titles', () => {
    render(<SearchResultsCard groups={mockGroups} total={2} query="test" />)
    expect(screen.getByText('alice')).toBeDefined()
    expect(screen.getByText('wiki-text')).toBeDefined()
  })

  it('shows no results message when total is 0', () => {
    render(<SearchResultsCard groups={[]} total={0} query="nothing" />)
    expect(screen.getByText(/No results for/)).toBeDefined()
  })

  it('renders item details', () => {
    render(<SearchResultsCard groups={mockGroups} total={2} query="test" />)
    expect(screen.getByText('alice@test.com')).toBeDefined()
  })

  it('renders result counts per group', () => {
    render(<SearchResultsCard groups={mockGroups} total={2} query="test" />)
    const counts = screen.getAllByText('(1)')
    expect(counts.length).toBe(2)
  })
})
