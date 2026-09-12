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
  EmptyCard: ({ message, action }: any) => (
    <div>{message}{action}</div>
  ),
  Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  StatusBanner: ({ message }: any) => <div>{message}</div>,

    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
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

vi.mock('@/components/icons/NavIcons', () => ({
  IconRefresh: (props: any) => <span data-testid="icon-refresh">refresh</span>,
}))

import { AgentRunsHistory } from './AgentRunsHistory'

afterEach(() => cleanup())

const mockRuns = [
  {
    id: 'run-1',
    status: 'completed',
    goal: 'Research transformers',
    response: 'Summary of findings',
    tasks: [
      { id: 't1', agent: 'Researcher', description: 'Search papers', status: 'completed' },
    ],
    completed_count: 1,
    failed_count: 0,
    started_at: '2026-09-10T10:00:00Z',
    finished_at: '2026-09-10T10:02:00Z',
    logs: ['started', 'done'],
  },
  {
    id: 'run-2',
    status: 'failed',
    goal: 'Build report',
    error: 'Agent timed out',
    tasks: [
      { id: 't2', agent: 'Coder', description: 'Write script', status: 'failed' },
    ],
    completed_count: 0,
    failed_count: 1,
    started_at: '2026-09-10T11:00:00Z',
    logs: [],
  },
  {
    id: 'run-3',
    status: 'running',
    goal: 'Generate analysis',
    tasks: [
      { id: 't3', agent: 'Analyst', description: 'Analyze data', status: 'completed' },
    ],
    completed_count: 1,
    failed_count: 0,
    started_at: '2026-09-10T12:00:00Z',
    logs: [],
  },
]

const defaultProps = {
  runs: mockRuns,
  loading: false,
  viewMode: 'list' as const,
  statusFilter: null,
  agentFilter: null,
  expandedRun: null,
  onRefresh: vi.fn(),
  onViewModeChange: vi.fn(),
  onStatusFilterChange: vi.fn(),
  onAgentFilterChange: vi.fn(),
  onExpandRun: vi.fn(),
}

describe('AgentRunsHistory', () => {
  it('renders Run History title', () => {
    render(<AgentRunsHistory {...defaultProps} />)
    expect(screen.getByText('Run History')).toBeTruthy()
  })

  it('shows loading skeletons', () => {
    render(<AgentRunsHistory {...defaultProps} loading={true} />)
    expect(screen.getByTestId('runs-loading')).toBeTruthy()
  })

  it('shows empty state', () => {
    render(<AgentRunsHistory {...defaultProps} runs={[]} />)
    expect(screen.getByText(/No runs yet/)).toBeTruthy()
  })

  it('renders run goals', () => {
    render(<AgentRunsHistory {...defaultProps} />)
    expect(screen.getByText('Research transformers')).toBeTruthy()
    expect(screen.getByText('Build report')).toBeTruthy()
  })

  it('shows filters when more than 2 runs', () => {
    render(<AgentRunsHistory {...defaultProps} />)
    expect(screen.getByTestId('runs-filters')).toBeTruthy()
  })

  it('shows status filter buttons', () => {
    render(<AgentRunsHistory {...defaultProps} />)
    expect(screen.getByText('All')).toBeTruthy()
    expect(screen.getAllByText('completed').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('failed').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onStatusFilterChange when filter clicked', () => {
    const onStatusFilterChange = vi.fn()
    render(<AgentRunsHistory {...defaultProps} onStatusFilterChange={onStatusFilterChange} />)
    const filterButtons = screen.getAllByText('completed')
    fireEvent.click(filterButtons[0])
    expect(onStatusFilterChange).toHaveBeenCalledWith('completed')
  })

  it('calls onViewModeChange when toggling view', () => {
    const onViewModeChange = vi.fn()
    render(<AgentRunsHistory {...defaultProps} onViewModeChange={onViewModeChange} />)
    fireEvent.click(screen.getByText('Timeline'))
    expect(onViewModeChange).toHaveBeenCalledWith('timeline')
  })

  it('expands a run on click', () => {
    const onExpandRun = vi.fn()
    render(<AgentRunsHistory {...defaultProps} onExpandRun={onExpandRun} />)
    fireEvent.click(screen.getByTestId('run-run-1'))
    expect(onExpandRun).toHaveBeenCalledWith('run-1')
  })

  it('shows expanded run details with tasks', () => {
    render(<AgentRunsHistory {...defaultProps} expandedRun="run-1" />)
    expect(screen.getByText('Search papers')).toBeTruthy()
    expect(screen.getByText('Result')).toBeTruthy()
  })

  it('shows error banner for failed runs', () => {
    render(<AgentRunsHistory {...defaultProps} expandedRun="run-2" />)
    expect(screen.getByText('Agent timed out')).toBeTruthy()
  })

  it('calls onRefresh on refresh click', () => {
    const onRefresh = vi.fn()
    render(<AgentRunsHistory {...defaultProps} onRefresh={onRefresh} />)
    fireEvent.click(screen.getByTestId('icon-refresh').closest('button')!)
    expect(onRefresh).toHaveBeenCalled()
  })
})
