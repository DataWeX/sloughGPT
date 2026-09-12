import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { KbSearchResults } from './KbSearchResults'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: ({ onChange, onKeyDown, ...props }: any) => (
    <input data-testid="input" onChange={onChange} onKeyDown={onKeyDown} {...props} />
  ),

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

describe('KbSearchResults', () => {
  const defaultProps = {
    query: '',
    results: [],
    loading: false,
    onQueryChange: vi.fn(),
    onSearch: vi.fn(),
  }

  it('renders the search input', () => {
    render(<KbSearchResults {...defaultProps} />)
    expect(screen.getByPlaceholderText('Search knowledge...')).toBeDefined()
  })

  it('renders the Search button', () => {
    render(<KbSearchResults {...defaultProps} />)
    expect(screen.getByText('Search')).toBeDefined()
  })

  it('displays results when provided', () => {
    const results = [
      { id: '1', content: 'Fact about cats', topic: 'animals', score: 0.95 },
      { id: '2', content: 'Fact about dogs', topic: 'animals', score: 0.87 },
    ]
    render(<KbSearchResults {...defaultProps} results={results} />)
    expect(screen.getByText('Fact about cats')).toBeDefined()
    expect(screen.getByText('Fact about dogs')).toBeDefined()
  })

  it('shows score for each result', () => {
    const results = [{ id: '1', content: 'Test fact', topic: 'test', score: 0.123 }]
    render(<KbSearchResults {...defaultProps} results={results} />)
    expect(screen.getByText('Score: 0.123')).toBeDefined()
  })

  it('shows topic badge for each result', () => {
    const results = [{ id: '1', content: 'Test', topic: 'science', score: 0.5 }]
    render(<KbSearchResults {...defaultProps} results={results} />)
    expect(screen.getByText('science')).toBeDefined()
  })

  it('shows Searching... when loading', () => {
    render(<KbSearchResults {...defaultProps} loading={true} />)
    expect(screen.getByText('Searching...')).toBeDefined()
  })

  it('calls onSearch when Search button clicked', () => {
    const onSearch = vi.fn()
    render(<KbSearchResults {...defaultProps} onSearch={onSearch} />)
    fireEvent.click(screen.getByText('Search'))
    expect(onSearch).toHaveBeenCalledOnce()
  })
})
