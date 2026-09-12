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
  Input: ({ ...props }: any) => <input {...props} />,
  StatusBanner: ({ message }: any) => <div>{message}</div>,

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

import { OrchestrationPanel } from './OrchestrationPanel'

afterEach(() => cleanup())

const mockAgents = [
  { id: 'a1', name: 'Researcher' },
  { id: 'a2', name: 'Coder' },
]

const defaultProps = {
  agents: mockAgents,
  goal: '',
  context: '',
  running: false,
  phase: '',
  tasks: [],
  taskStatuses: {},
  level: 0,
  totalLevels: 0,
  response: null,
  error: null,
  selectedAgentIds: [],
  errors: {},
  onGoalChange: vi.fn(),
  onContextChange: vi.fn(),
  onAgentToggle: vi.fn(),
  onOrchestrate: vi.fn(),
  onClear: vi.fn(),
}

describe('OrchestrationPanel', () => {
  it('renders title and description', () => {
    render(<OrchestrationPanel {...defaultProps} />)
    expect(screen.getByText('Multi-Agent Orchestration')).toBeTruthy()
    expect(screen.getByText(/Decompose a goal/)).toBeTruthy()
  })

  it('renders agent toggle buttons', () => {
    render(<OrchestrationPanel {...defaultProps} />)
    expect(screen.getByText('Researcher')).toBeTruthy()
    expect(screen.getByText('Coder')).toBeTruthy()
  })

  it('calls onAgentToggle when agent clicked', () => {
    const onAgentToggle = vi.fn()
    render(<OrchestrationPanel {...defaultProps} onAgentToggle={onAgentToggle} />)
    fireEvent.click(screen.getByText('Researcher'))
    expect(onAgentToggle).toHaveBeenCalledWith('a1')
  })

  it('calls onOrchestrate when button clicked', () => {
    const onOrchestrate = vi.fn()
    render(<OrchestrationPanel {...defaultProps} goal="test" onOrchestrate={onOrchestrate} />)
    fireEvent.click(screen.getByText('Orchestrate'))
    expect(onOrchestrate).toHaveBeenCalled()
  })

  it('disables Orchestrate when goal empty', () => {
    render(<OrchestrationPanel {...defaultProps} goal="" />)
    expect(screen.getByText('Orchestrate').hasAttribute('disabled')).toBe(true)
  })

  it('shows Orchestrating... when running', () => {
    render(<OrchestrationPanel {...defaultProps} running={true} goal="test" />)
    expect(screen.getByText('Orchestrating...')).toBeTruthy()
  })

  it('shows phase indicator', () => {
    render(<OrchestrationPanel {...defaultProps} phase="PLAN" />)
    expect(screen.getByText('Planning subtasks...')).toBeTruthy()
    expect(screen.getByTestId('phase-indicator')).toBeTruthy()
  })

  it('shows tasks list', () => {
    const tasks = [
      { id: 't1', agent: 'Researcher', description: 'Find papers' },
      { id: 't2', agent: 'Coder', description: 'Write script', depends_on: ['t1'] },
    ]
    render(<OrchestrationPanel {...defaultProps} tasks={tasks} />)
    expect(screen.getByText('Find papers')).toBeTruthy()
    expect(screen.getByText('Write script')).toBeTruthy()
    expect(screen.getByText('after: t1')).toBeTruthy()
  })

  it('shows response result', () => {
    render(<OrchestrationPanel {...defaultProps} response="Final answer here" />)
    expect(screen.getByText('Result')).toBeTruthy()
    expect(screen.getByText('Final answer here')).toBeTruthy()
  })

  it('shows error banner', () => {
    render(<OrchestrationPanel {...defaultProps} error="Something failed" />)
    expect(screen.getByText('Something failed')).toBeTruthy()
  })

  it('calls onClear when Clear clicked', () => {
    const onClear = vi.fn()
    render(<OrchestrationPanel {...defaultProps} response="done" onClear={onClear} />)
    fireEvent.click(screen.getByText('Clear'))
    expect(onClear).toHaveBeenCalled()
  })

  it('shows goal validation error', () => {
    render(<OrchestrationPanel {...defaultProps} errors={{ goal: 'Goal required' }} />)
    expect(screen.getByText('Goal required')).toBeTruthy()
    expect(screen.getByRole('alert')).toBeTruthy()
  })
})
