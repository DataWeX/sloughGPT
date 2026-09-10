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
