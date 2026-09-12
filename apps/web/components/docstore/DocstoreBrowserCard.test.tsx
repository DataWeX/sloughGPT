// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: ({ value, onChange, ...props }: any) => (
    <input value={value} onChange={onChange} {...props} />
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

import { DocstoreBrowserCard } from './DocstoreBrowserCard'

afterEach(() => cleanup())

const defaultProps = {
  collections: ['default', 'code'],
  selected: 'default',
  docs: [{ _id: 'doc-1' }, { _id: 'doc-2' }],
  page: 1,
  totalPages: 3,
  loading: false,
  searchQuery: '',
  onSelect: vi.fn(),
  onSearchChange: vi.fn(),
  onPageChange: vi.fn(),
  onCreateToggle: vi.fn(),
  showCreate: false,
}

describe('DocstoreBrowserCard', () => {
  it('renders the title', () => {
    render(<DocstoreBrowserCard {...defaultProps} />)
    expect(screen.getByText('Documents')).toBeTruthy()
  })

  it('renders collection tabs', () => {
    render(<DocstoreBrowserCard {...defaultProps} />)
    expect(screen.getByTestId('collection-default')).toBeTruthy()
    expect(screen.getByTestId('collection-code')).toBeTruthy()
    expect(screen.getByText('default')).toBeTruthy()
    expect(screen.getByText('code')).toBeTruthy()
  })

  it('renders New doc button', () => {
    render(<DocstoreBrowserCard {...defaultProps} />)
    expect(screen.getByTestId('new-doc-btn')).toBeTruthy()
    expect(screen.getByText('New doc')).toBeTruthy()
  })

  it('shows document list when docs provided', () => {
    render(<DocstoreBrowserCard {...defaultProps} />)
    expect(screen.getByTestId('doc-list')).toBeTruthy()
    expect(screen.getByText('doc-1')).toBeTruthy()
    expect(screen.getByText('doc-2')).toBeTruthy()
  })

  it('calls onSelect when a doc is clicked', () => {
    const onSelect = vi.fn()
    render(<DocstoreBrowserCard {...defaultProps} onSelect={onSelect} />)
    fireEvent.click(screen.getByTestId('doc-item-doc-1'))
    expect(onSelect).toHaveBeenCalledWith('doc-1')
  })

  it('calls onSearchChange when typing', () => {
    const onSearchChange = vi.fn()
    render(<DocstoreBrowserCard {...defaultProps} onSearchChange={onSearchChange} />)
    fireEvent.change(screen.getByTestId('doc-search'), { target: { value: 'test' } })
    expect(onSearchChange).toHaveBeenCalledWith('test')
  })

  it('calls onCreateToggle when New doc clicked', () => {
    const onCreateToggle = vi.fn()
    render(<DocstoreBrowserCard {...defaultProps} onCreateToggle={onCreateToggle} />)
    fireEvent.click(screen.getByTestId('new-doc-btn'))
    expect(onCreateToggle).toHaveBeenCalled()
  })

  it('shows pagination when totalPages > 1', () => {
    render(<DocstoreBrowserCard {...defaultProps} />)
    expect(screen.getByTestId('pagination')).toBeTruthy()
    expect(screen.getByText('1 / 3')).toBeTruthy()
  })

  it('disables Prev on first page', () => {
    render(<DocstoreBrowserCard {...defaultProps} page={1} />)
    const prevBtn = screen.getByText('Prev')
    expect(prevBtn).toHaveProperty('disabled', true)
  })

  it('disables Next on last page', () => {
    render(<DocstoreBrowserCard {...defaultProps} page={3} totalPages={3} />)
    const nextBtn = screen.getByText('Next')
    expect(nextBtn).toHaveProperty('disabled', true)
  })

  it('calls onPageChange when navigating', () => {
    const onPageChange = vi.fn()
    render(<DocstoreBrowserCard {...defaultProps} onPageChange={onPageChange} />)
    fireEvent.click(screen.getByText('Next'))
    expect(onPageChange).toHaveBeenCalledWith(2)
  })

  it('shows loading state', () => {
    render(<DocstoreBrowserCard {...defaultProps} loading={true} docs={[]} />)
    expect(screen.getByTestId('loading-indicator')).toBeTruthy()
    expect(screen.getByText('Loading...')).toBeTruthy()
  })

  it('shows empty state when no docs', () => {
    render(<DocstoreBrowserCard {...defaultProps} docs={[]} />)
    expect(screen.getByText('No documents found.')).toBeTruthy()
  })

  it('shows Cancel when showCreate is true', () => {
    render(<DocstoreBrowserCard {...defaultProps} showCreate={true} />)
    expect(screen.getByText('Cancel')).toBeTruthy()
  })
})
