import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('class-variance-authority', () => ({ cva: () => () => '' }))

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; C.displayName = `Icon${name}`; return C }
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Button: ({ children, onClick, variant, size, className, disabled, 'aria-label': ariaLabel }: any) => (
      <button onClick={onClick} className={className} disabled={disabled} aria-label={ariaLabel} data-variant={variant}>{children}</button>
    ),
    Input: ({ value, onChange, className, placeholder }: any) => <input value={value} onChange={onChange} className={className} placeholder={placeholder} />,
    EmptyCard: ({ message }: any) => <div>{message}</div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    StatCard: ({ label, value }: any) => <div><span>{label}</span><span>{String(value)}</span></div>,
    IconRefresh: iconMock('refresh'), IconPlus: iconMock('plus'), IconTrash: iconMock('trash'), IconClock: iconMock('clock'),
    AlertDialog: ({ children, open, onOpenChange }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
    AlertDialogContent: ({ children }: any) => <div>{children}</div>,
    AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
    AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
    AlertDialogDescription: ({ children }: any) => <div>{children}</div>,
    AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
    AlertDialogCancel: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    AlertDialogAction: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  }
})

vi.mock('@/components/icons/NavIcons', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; C.displayName = `Icon${name}`; return C }
  return { IconPlus: iconMock('plus'), IconTrash: iconMock('trash'), IconClock: iconMock('clock') }
})

vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ left }: any) => <header>{left}</header>,
  AppRouteHeaderLead: ({ title }: any) => <h1>{title}</h1>,
}))

const { mockList, mockListRuns, mockAddToast } = vi.hoisted(() => ({
  mockList: vi.fn(), mockListRuns: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('@/lib/agents-controller', () => ({
  agentsController: { list: mockList, listRuns: mockListRuns, create: vi.fn(), update: vi.fn(), delete: vi.fn(), execute: vi.fn(), orchestrate: vi.fn() },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

import AgentsPage from './page'

const RUN_FIXTURE = {
  id: 'run_1',
  goal: 'Research transformers',
  context: '',
  status: 'completed',
  started_at: '2026-08-01T12:00:00Z',
  finished_at: '2026-08-01T12:01:00Z',
  tasks: [
    { id: 't1', description: 'Gather papers', agent: 'researcher', status: 'completed', result_preview: 'notes', depends_on: [] },
    { id: 't2', description: 'Write summary', agent: 'writer', status: 'completed', result_preview: 'summary', depends_on: ['t1'] },
  ],
  completed_count: 2,
  failed_count: 0,
  response: 'Here is the summary.',
  error: '',
  logs: ['[2026-08-01T12:00:00Z] Started: Research transformers', '[2026-08-01T12:01:00Z] Completed'],
}

describe('AgentsPage Run History', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue([])
    mockListRuns.mockResolvedValue({ runs: [], count: 0 })
  })

  afterEach(() => cleanup())

  it('shows empty run history state', async () => {
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getByText('Run History')).toBeTruthy() })
    await waitFor(() => { expect(screen.getByText(/No runs yet/)).toBeTruthy() })
  })

  it('lists runs with status and goal', async () => {
    mockListRuns.mockResolvedValue({ runs: [RUN_FIXTURE], count: 1 })
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getByText('Research transformers')).toBeTruthy() })
    expect(mockListRuns).toHaveBeenCalledWith(20)
    expect(screen.getByText('2/2 tasks')).toBeTruthy()
  })

  it('expands a run to show tasks, result, and logs', async () => {
    mockListRuns.mockResolvedValue({ runs: [RUN_FIXTURE], count: 1 })
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getByText('Research transformers')).toBeTruthy() })
    screen.getByText('Research transformers').click()
    await waitFor(() => { expect(screen.getByText('Gather papers')).toBeTruthy() })
    expect(screen.getByText('Write summary')).toBeTruthy()
    expect(screen.getByText('Here is the summary.')).toBeTruthy()
    expect(screen.getByText(/Started: Research transformers/)).toBeTruthy()
  })

  it('shows error message for failed runs', async () => {
    mockListRuns.mockResolvedValue({
      runs: [{ ...RUN_FIXTURE, id: 'run_2', status: 'failed', error: 'LLM failed', response: '', tasks: [] }],
      count: 1,
    })
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getByText('Research transformers')).toBeTruthy() })
    screen.getByText('Research transformers').click()
    await waitFor(() => { expect(screen.getByText('LLM failed')).toBeTruthy() })
  })

  it('handles run history fetch failure gracefully', async () => {
    mockListRuns.mockRejectedValue(new Error('500'))
    render(<AgentsPage />)
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Failed to load run history', 'error') })
  })

  it('calls listRuns with default limit on mount', async () => {
    render(<AgentsPage />)
    await waitFor(() => { expect(mockListRuns).toHaveBeenCalledWith(20) })
  })
})
