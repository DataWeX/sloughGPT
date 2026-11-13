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
  Checkbox: ({ checked, onCheckedChange, ...props }: any) => (
    <input
      type="checkbox"
      checked={!!checked}
      onChange={() => onCheckedChange?.(!checked)}
      {...props}
    />
  ),
  EmptyCard: ({ message }: any) => <div>{message}</div>,
  Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,

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
  IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
}))

import { AgentListToolbar } from './AgentListToolbar'

afterEach(() => cleanup())

const mockAgents = [
  { id: 'a1', name: 'Researcher', description: 'Finds info', tools: ['web_search'], instructions: 'Be thorough' },
  { id: 'a2', name: 'Coder', description: 'Writes code', tools: ['code_execution'], instructions: 'Write clean' },
  { id: 'a3', name: 'Analyst', description: 'Analyzes data', tools: ['data_analysis'] },
]

const defaultProps = {
  agents: mockAgents,
  loading: false,
  selectedIds: new Set<string>(),
  search: '',
  onSearchChange: vi.fn(),
  onSelectAll: vi.fn(),
  onRefresh: vi.fn(),
  onBulkExport: vi.fn(),
  onBulkDelete: vi.fn(),
  renderAgent: (agent: any) => <div key={agent.id} data-testid={`agent-${agent.id}`}>{agent.name}</div>,
}

describe('AgentListToolbar', () => {
  it('renders Agents title', () => {
    render(<AgentListToolbar {...defaultProps} />)
    expect(screen.getByText('Agents')).toBeTruthy()
  })

  it('shows loading skeletons', () => {
    render(<AgentListToolbar {...defaultProps} loading={true} />)
    expect(screen.getByTestId('agents-loading')).toBeTruthy()
  })

  it('shows empty state', () => {
    render(<AgentListToolbar {...defaultProps} agents={[]} />)
    expect(screen.getByText('No agents yet')).toBeTruthy()
  })

  it('renders agent list via renderAgent', () => {
    render(<AgentListToolbar {...defaultProps} />)
    expect(screen.getByTestId('agent-a1')).toBeTruthy()
    expect(screen.getByTestId('agent-a2')).toBeTruthy()
    expect(screen.getByTestId('agent-a3')).toBeTruthy()
  })

  it('shows search input when more than 2 agents', () => {
    render(<AgentListToolbar {...defaultProps} />)
    expect(screen.getByPlaceholderText('Search agents...')).toBeTruthy()
  })

  it('calls onSearchChange when searching', () => {
    const onSearchChange = vi.fn()
    render(<AgentListToolbar {...defaultProps} onSearchChange={onSearchChange} />)
    fireEvent.change(screen.getByPlaceholderText('Search agents...'), { target: { value: 'code' } })
    expect(onSearchChange).toHaveBeenCalledWith('code')
  })

  it('shows select all checkbox', () => {
    render(<AgentListToolbar {...defaultProps} />)
    expect(screen.getByLabelText('Select all agents')).toBeTruthy()
  })

  it('calls onSelectAll when checkbox clicked', () => {
    const onSelectAll = vi.fn()
    render(<AgentListToolbar {...defaultProps} onSelectAll={onSelectAll} />)
    fireEvent.click(screen.getByLabelText('Select all agents'))
    expect(onSelectAll).toHaveBeenCalled()
  })

  it('shows bulk actions when items selected', () => {
    render(<AgentListToolbar {...defaultProps} selectedIds={new Set(['a1', 'a2'])} />)
    expect(screen.getByText('2 selected')).toBeTruthy()
    expect(screen.getByText('Export')).toBeTruthy()
    expect(screen.getByText('Delete')).toBeTruthy()
  })

  it('calls onBulkDelete when delete clicked', () => {
    const onBulkDelete = vi.fn()
    render(<AgentListToolbar {...defaultProps} selectedIds={new Set(['a1'])} onBulkDelete={onBulkDelete} />)
    fireEvent.click(screen.getByText('Delete'))
    expect(onBulkDelete).toHaveBeenCalled()
  })

  it('calls onBulkExport when export clicked', () => {
    const onBulkExport = vi.fn()
    render(<AgentListToolbar {...defaultProps} selectedIds={new Set(['a1'])} onBulkExport={onBulkExport} />)
    fireEvent.click(screen.getByText('Export'))
    expect(onBulkExport).toHaveBeenCalled()
  })

  it('shows no match message', () => {
    render(<AgentListToolbar {...defaultProps} search="zzz" />)
    expect(screen.getByText(/No agents matching/)).toBeTruthy()
  })

  it('calls onRefresh on refresh click', () => {
    const onRefresh = vi.fn()
    render(<AgentListToolbar {...defaultProps} onRefresh={onRefresh} />)
    fireEvent.click(screen.getByTestId('icon-refresh').closest('button')!)
    expect(onRefresh).toHaveBeenCalled()
  })
})
