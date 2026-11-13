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
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,

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

vi.mock('@/lib/time-ago', () => ({
  timeAgo: () => '2d ago',
}))

import { CollectionPipelineCard } from './CollectionPipelineCard'

afterEach(() => cleanup())

const mockPipelines = [
  { id: 'p-1', name: 'RSS Loader', source_type: 'rss', store_type: 'memory', records_count: 120, last_run: '2026-09-08' },
  { id: 'p-2', name: 'File Import', source_type: 'file', store_type: 'file', records_count: 45 },
  { id: 'p-3', name: 'API Sync', source_type: 'api', store_type: 'chained' },
]

describe('CollectionPipelineCard', () => {
  it('renders nothing when empty', () => {
    const { container } = render(<CollectionPipelineCard pipelines={[]} />)
    expect(container.firstChild).toBeNull()
  })

  it('shows pipeline list', () => {
    render(<CollectionPipelineCard pipelines={mockPipelines} />)
    expect(screen.getByText('Pipelines')).toBeTruthy()
    expect(screen.getByText('(3)')).toBeTruthy()
    expect(screen.getByText('RSS Loader')).toBeTruthy()
    expect(screen.getByText('File Import')).toBeTruthy()
    expect(screen.getByText('API Sync')).toBeTruthy()
  })

  it('shows source type badges', () => {
    render(<CollectionPipelineCard pipelines={mockPipelines} />)
    expect(screen.getByText('rss')).toBeTruthy()
    expect(screen.getByText('file')).toBeTruthy()
    expect(screen.getByText('api')).toBeTruthy()
  })

  it('shows store types', () => {
    render(<CollectionPipelineCard pipelines={mockPipelines} />)
    expect(screen.getByText('→ memory')).toBeTruthy()
    expect(screen.getByText('→ file')).toBeTruthy()
    expect(screen.getByText('→ chained')).toBeTruthy()
  })

  it('shows record counts', () => {
    render(<CollectionPipelineCard pipelines={mockPipelines} />)
    expect(screen.getByText('120 records')).toBeTruthy()
    expect(screen.getByText('45 records')).toBeTruthy()
  })

  it('calls onRun', () => {
    const onRun = vi.fn()
    render(<CollectionPipelineCard pipelines={mockPipelines} onRun={onRun} />)
    const runBtns = screen.getAllByRole('button').filter(b => b.textContent === 'Run')
    fireEvent.click(runBtns[0])
    expect(onRun).toHaveBeenCalledWith('p-1')
  })

  it('calls onDelete', () => {
    const onDelete = vi.fn()
    render(<CollectionPipelineCard pipelines={mockPipelines} onDelete={onDelete} />)
    const delBtns = screen.getAllByRole('button').filter(b => b.textContent === 'Del')
    fireEvent.click(delBtns[0])
    expect(onDelete).toHaveBeenCalledWith('p-1')
  })

  it('shows running state', () => {
    render(<CollectionPipelineCard pipelines={mockPipelines} runningId="p-2" onRun={vi.fn()} />)
    const runningBtn = screen.getAllByRole('button').find(b => b.textContent === '...')
    expect(runningBtn).toBeTruthy()
    expect(runningBtn!.hasAttribute('disabled')).toBe(true)
  })
})
