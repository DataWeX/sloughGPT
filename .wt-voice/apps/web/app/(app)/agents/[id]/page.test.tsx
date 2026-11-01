import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; C.displayName = `Icon${name}`; return C }
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Button: ({ children, onClick, disabled, 'aria-label': ariaLabel, className }: any) => (
      <button onClick={onClick} disabled={disabled} aria-label={ariaLabel} className={className}>{children}</button>
    ),
    Badge: ({ children, variant }: any) => <span data-variant={variant}>{children}</span>,
    Input: ({ value, onChange, placeholder, className, onKeyDown }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} className={className} onKeyDown={onKeyDown} data-testid={placeholder === 'Enter a prompt to execute...' ? 'exec-input' : undefined} />
    ),
    Skeleton: () => <div data-testid="skeleton" />,
    IconRefresh: iconMock('refresh'),
  }
})

vi.mock('lucide-react', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`lucide-${name}`}>{name}</span>; C.displayName = `Lucide${name}`; return C }
  return {
    ArrowLeft: iconMock('arrow-left'), Bot: iconMock('bot'), Wrench: iconMock('wrench'),
    FileText: iconMock('file-text'), Play: iconMock('play'), History: iconMock('history'),
    Edit: iconMock('edit'), Trash2: iconMock('trash-2'), Loader2: iconMock('loader-2'),
    CheckCircle: iconMock('check-circle'), XCircle: iconMock('x-circle'),
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, loading, loadingCards, headerRight }: any) => (
    <div data-testid="page-container" data-title={title} data-subtitle={subtitle} data-loading={loading}>
      <div data-testid="header-right">{headerRight}</div>
      {loading ? <div data-testid="loading-state" /> : children}
    </div>
  ),
}))

vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ children }: any) => <div>{children}</div>,
  AppRouteHeaderLead: ({ children }: any) => <span>{children}</span>,
}))

const { mockList, mockListRuns, mockUpdate, mockDelete, mockExecute, mockPush, mockAddToast } = vi.hoisted(() => ({
  mockList: vi.fn(), mockListRuns: vi.fn(), mockUpdate: vi.fn(),
  mockDelete: vi.fn(), mockExecute: vi.fn(), mockPush: vi.fn(), mockAddToast: vi.fn(),
}))

const stableRouter = { push: mockPush }
vi.mock('next/navigation', () => ({
  useParams: () => ({ id: 'agent-1' }),
  useRouter: () => stableRouter,
}))

vi.mock('@/lib/agents-controller', () => ({
  agentsController: {
    list: mockList, listRuns: mockListRuns, update: mockUpdate,
    delete: mockDelete, execute: mockExecute,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

import Page from './page'

const MOCK_AGENT: any = {
  id: 'agent-1',
  name: 'Test Agent',
  description: 'A test agent',
  instructions: 'Be helpful',
  tools: ['code_execution', 'web_search'],
  avatar: '',
}

const MOCK_RUNS: any = {
  runs: [
    {
      id: 'run-1',
      goal: 'Solve a problem',
      status: 'completed',
      tasks: [{ agent: 'agent-1', id: 't1', description: 'do thing', status: 'completed', result_preview: '', depends_on: [] }],
      completed_count: 1,
      failed_count: 0,
    },
  ],
  count: 1,
}

afterEach(() => cleanup())
beforeEach(() => {
  vi.clearAllMocks()
  mockList.mockResolvedValue([MOCK_AGENT])
  mockListRuns.mockResolvedValue(MOCK_RUNS)
  mockUpdate.mockResolvedValue(MOCK_AGENT)
  mockDelete.mockResolvedValue(undefined)
  mockExecute.mockResolvedValue({ response: 'Execution result', tools_used: [] })
})

describe('AgentDetailPage', () => {
  it('shows loading state on mount', () => {
    mockList.mockReturnValue(new Promise(() => {}))
    render(<Page />)
    expect(screen.getByTestId('page-container')).toHaveAttribute('data-loading', 'true')
  })

  it('renders agent name after loading', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByTestId('page-container')).toHaveAttribute('data-title', 'Test Agent')
    })
  })

  it('displays agent info card with name, description, and id', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Test Agent')).toBeTruthy() })
    expect(screen.getByText('A test agent')).toBeTruthy()
    expect(screen.getByText('agent-1')).toBeTruthy()
    expect(screen.getByText('Agent Info')).toBeTruthy()
  })

  it('displays tools card with assigned tools', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Tools')).toBeTruthy() })
    expect(screen.getByText('Code Execution')).toBeTruthy()
    expect(screen.getByText('Web Search')).toBeTruthy()
  })

  it('displays instructions card', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Instructions')).toBeTruthy() })
    expect(screen.getByText('Be helpful')).toBeTruthy()
  })

  it('displays execute card with run button', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Execute')).toBeTruthy() })
    expect(screen.getByText('Run')).toBeTruthy()
  })

  it('displays recent runs card when runs exist', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Recent Runs')).toBeTruthy() })
    expect(screen.getByText('Solve a problem')).toBeTruthy()
    expect(screen.getByText('1/1 tasks')).toBeTruthy()
  })

  it('hides recent runs when no runs match agent', async () => {
    mockListRuns.mockResolvedValue({ runs: [{ id: 'run-x', goal: 'Other', tasks: [{ agent: 'other-agent', id: 't2', description: '', status: 'completed', result_preview: '', depends_on: [] }], completed_count: 1, failed_count: 0 }], count: 1 })
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Test Agent')).toBeTruthy() })
    expect(screen.queryByText('Recent Runs')).toBeNull()
  })

  it('navigates to /agents on Back button click', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Back')).toBeTruthy() })
    await act(async () => { screen.getByText('Back').click() })
    expect(mockPush).toHaveBeenCalledWith('/agents')
  })

  it('refetches agent data on refresh click', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Test Agent')).toBeTruthy() })
    await act(async () => { screen.getByTestId('icon-refresh').click() })
    await waitFor(() => { expect(mockList).toHaveBeenCalledTimes(2) })
  })

  it('enters edit mode and saves changes', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Edit')).toBeTruthy() })
    await act(async () => { screen.getByText('Edit').click() })

    expect(screen.getByText('Cancel')).toBeTruthy()
    expect(screen.getByText('Save')).toBeTruthy()

    await act(async () => { screen.getByText('Save').click() })
    await waitFor(() => { expect(mockUpdate).toHaveBeenCalled() })
    expect(mockAddToast).toHaveBeenCalledWith('Updated', 'success')
  })

  it('cancels edit mode without saving', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Edit')).toBeTruthy() })
    await act(async () => { screen.getByText('Edit').click() })
    await act(async () => { screen.getByText('Cancel').click() })
    expect(screen.getByText('Edit')).toBeTruthy()
    expect(mockUpdate).not.toHaveBeenCalled()
  })

  it('deletes agent and navigates to /agents', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Delete')).toBeTruthy() })
    await act(async () => { screen.getByText('Delete').click() })
    await waitFor(() => { expect(mockDelete).toHaveBeenCalledWith('agent-1') })
    expect(mockAddToast).toHaveBeenCalledWith('Deleted', 'success')
    expect(mockPush).toHaveBeenCalledWith('/agents')
  })

  it('handles delete error', async () => {
    mockDelete.mockRejectedValue(new Error('fail'))
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Delete')).toBeTruthy() })
    await act(async () => { screen.getByText('Delete').click() })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Delete failed', 'error') })
  })

  it('executes agent with prompt and displays result', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Execute')).toBeTruthy() })
    const input = screen.getByTestId('exec-input')
    await act(async () => { fireEvent.change(input, { target: { value: 'What is 2+2?' } }) })
    await act(async () => { screen.getByText('Run').click() })
    await waitFor(() => { expect(mockExecute).toHaveBeenCalledWith('agent-1', 'What is 2+2?') })
    expect(screen.getByText('Execution result')).toBeTruthy()
  })

  it('handles execution error', async () => {
    mockExecute.mockRejectedValue(new Error('fail'))
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Execute')).toBeTruthy() })
    const input = screen.getByTestId('exec-input')
    await act(async () => { fireEvent.change(input, { target: { value: 'test' } }) })
    await act(async () => { screen.getByText('Run').click() })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Execution failed', 'error') })
  })

  it('navigates to chat with agent', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Chat with Agent')).toBeTruthy() })
    await act(async () => { screen.getByText('Chat with Agent').click() })
    expect(mockPush).toHaveBeenCalledWith('/chat?agent=agent-1')
  })

  it('handles fetch error and shows toast', async () => {
    mockList.mockRejectedValue(new Error('network error'))
    render(<Page />)
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Failed to load agent', 'error') })
  })

  it('redirects to /agents when agent not found', async () => {
    mockList.mockResolvedValue([])
    render(<Page />)
    await waitFor(() => { expect(mockPush).toHaveBeenCalledWith('/agents') })
  })

  it('shows no tools assigned when agent has empty tools', async () => {
    mockList.mockResolvedValue([{ ...MOCK_AGENT, tools: [] }])
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('No tools assigned')).toBeTruthy() })
  })

  it('shows no instructions set when agent has empty instructions', async () => {
    mockList.mockResolvedValue([{ ...MOCK_AGENT, instructions: '' }])
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('No instructions set')).toBeTruthy() })
  })

  it('shows dash for agent with no description', async () => {
    mockList.mockResolvedValue([{ ...MOCK_AGENT, description: '' }])
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('\u2014')).toBeTruthy() })
  })
})
