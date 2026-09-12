// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
  Input: ({ ...props }: any) => <input {...props} />,

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

vi.mock('lucide-react', () => ({
  Search: () => null,
  Trash2: () => null,
  ArrowUpDown: () => null,
}))

import { DatasetListCard } from './DatasetListCard'

afterEach(() => cleanup())

const mockDatasets = [
  { id: '1', name: 'train.jsonl', size: 1048576, row_count: 5000, updated_at: '2025-01-10', tags: ['train'] },
  { id: '2', name: 'eval.jsonl', size: 262144, row_count: 1000, updated_at: '2025-01-12' },
]

describe('DatasetListCard', () => {
  const defaultProps = {
    datasets: mockDatasets,
    loading: false,
    search: '',
    sortBy: 'date' as const,
    onSearchChange: vi.fn(),
    onSortChange: vi.fn(),
    onSelect: vi.fn(),
    onDelete: vi.fn(),
  }

  it('renders title', () => {
    render(<DatasetListCard {...defaultProps} />)
    expect(screen.getByText('Datasets')).toBeTruthy()
  })

  it('renders search input', () => {
    render(<DatasetListCard {...defaultProps} />)
    expect(screen.getByTestId('dataset-search')).toBeTruthy()
  })

  it('renders sort buttons', () => {
    render(<DatasetListCard {...defaultProps} />)
    expect(screen.getByText('Date')).toBeTruthy()
    expect(screen.getByText('Size')).toBeTruthy()
    expect(screen.getByText('Name')).toBeTruthy()
  })

  it('renders dataset entries when provided', () => {
    render(<DatasetListCard {...defaultProps} />)
    expect(screen.getByText('train.jsonl')).toBeTruthy()
    expect(screen.getByText('eval.jsonl')).toBeTruthy()
    expect(screen.getByText('5,000 rows')).toBeTruthy()
    expect(screen.getByText('1,000 rows')).toBeTruthy()
  })

  it('shows loading placeholders', () => {
    render(<DatasetListCard {...defaultProps} loading={true} datasets={[]} />)
    expect(screen.queryByText('train.jsonl')).toBeNull()
  })

  it('shows empty message', () => {
    render(<DatasetListCard {...defaultProps} datasets={[]} />)
    expect(screen.getByText('No datasets found.')).toBeTruthy()
  })

  it('renders tags when present', () => {
    render(<DatasetListCard {...defaultProps} />)
    expect(screen.getByText('train')).toBeTruthy()
  })
})
