import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { SimilaritySearchCard } from './SimilaritySearchCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: (props: any) => <input data-testid="search-input" {...props} />,

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

const defaultProps = {
  searchQuery: '',
  searchResults: [],
  searchTime: null,
  searching: false,
  onQueryChange: vi.fn(),
  onSearch: vi.fn(),
}

describe('SimilaritySearchCard', () => {
  it('renders the card title', () => {
    render(<SimilaritySearchCard {...defaultProps} />)
    expect(screen.getByText('Similarity Search')).toBeDefined()
  })

  it('renders the search input', () => {
    render(<SimilaritySearchCard {...defaultProps} />)
    expect(screen.getByTestId('search-input')).toBeDefined()
  })

  it('renders the search button', () => {
    render(<SimilaritySearchCard {...defaultProps} />)
    expect(screen.getByText('Search')).toBeDefined()
  })

  it('displays search results with text and score', () => {
    const results = [
      { id: '1', text: 'Hello world', score: 0.95 },
      { id: '2', text: 'Test entry', score: 0.82 },
    ]
    render(<SimilaritySearchCard {...defaultProps} searchResults={results} searchTime={12.5} />)
    expect(screen.getByText('Hello world')).toBeDefined()
    expect(screen.getByText('Test entry')).toBeDefined()
    expect(screen.getByText('95.0%')).toBeDefined()
    expect(screen.getByText('82.0%')).toBeDefined()
  })

  it('shows result count and time', () => {
    const results = [{ id: '1', text: 'Item', score: 0.9 }]
    render(<SimilaritySearchCard {...defaultProps} searchResults={results} searchTime={5.2} />)
    expect(screen.getByText('1 results in 5.2ms')).toBeDefined()
  })

  it('shows no results message when empty', () => {
    render(<SimilaritySearchCard {...defaultProps} searchQuery="test" />)
    expect(screen.getByText('No results found')).toBeDefined()
  })

  it('shows searching state', () => {
    render(<SimilaritySearchCard {...defaultProps} searching={true} />)
    expect(screen.getByText('Searching...')).toBeDefined()
  })
})
