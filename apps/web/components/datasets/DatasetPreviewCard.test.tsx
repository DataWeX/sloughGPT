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
}))

import { DatasetPreviewCard } from './DatasetPreviewCard'

afterEach(() => cleanup())

describe('DatasetPreviewCard', () => {
  const defaultProps = {
    datasetName: 'train.jsonl',
    preview: null,
    loading: false,
    search: '',
    onSearchChange: vi.fn(),
  }

  it('renders title with dataset name', () => {
    render(<DatasetPreviewCard {...defaultProps} />)
    expect(screen.getByText('Preview: train.jsonl')).toBeTruthy()
  })

  it('renders search input', () => {
    render(<DatasetPreviewCard {...defaultProps} />)
    expect(screen.getByTestId('preview-search')).toBeTruthy()
  })

  it('shows no data message when preview is null', () => {
    render(<DatasetPreviewCard {...defaultProps} />)
    expect(screen.getByText('No preview data available.')).toBeTruthy()
  })

  it('renders table with column headers', () => {
    const preview = { columns: ['id', 'text', 'label'], rows: [['1', 'hello', 'pos']] }
    render(<DatasetPreviewCard {...defaultProps} preview={preview} />)
    expect(screen.getByText('id')).toBeTruthy()
    expect(screen.getByText('text')).toBeTruthy()
    expect(screen.getByText('label')).toBeTruthy()
  })

  it('renders table rows', () => {
    const preview = {
      columns: ['name', 'value'],
      rows: [['alpha', '100'], ['beta', '200']],
    }
    render(<DatasetPreviewCard {...defaultProps} preview={preview} />)
    expect(screen.getByText('alpha')).toBeTruthy()
    expect(screen.getByText('100')).toBeTruthy()
    expect(screen.getByText('beta')).toBeTruthy()
    expect(screen.getByText('200')).toBeTruthy()
  })

  it('shows loading state', () => {
    render(<DatasetPreviewCard {...defaultProps} loading={true} />)
    expect(screen.queryByText('No preview data available.')).toBeNull()
  })

  it('renders table element', () => {
    const preview = { columns: ['col'], rows: [['val']] }
    render(<DatasetPreviewCard {...defaultProps} preview={preview} />)
    expect(screen.getByTestId('preview-table')).toBeTruthy()
  })
})
