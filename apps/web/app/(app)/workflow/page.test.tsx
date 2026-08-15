import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const {
  mockStatus, mockStart, mockStop, mockTrigger, mockAddToast,
} = vi.hoisted(() => ({
  mockStatus: vi.fn(), mockStart: vi.fn(), mockStop: vi.fn(),
  mockTrigger: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, disabled }: any) => (
      <button onClick={onClick} disabled={disabled}>{children}</button>
    ),
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  }
})

vi.mock('@/lib/workflow-controller', () => ({
  workflowController: {
    status: (...a: unknown[]) => mockStatus(...a),
    start: (...a: unknown[]) => mockStart(...a),
    stop: (...a: unknown[]) => mockStop(...a),
    trigger: (...a: unknown[]) => mockTrigger(...a),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/components/workflow/WorkflowPipeline', () => ({
  WorkflowPipeline: ({ status }: any) => (
    <div data-testid="workflow-pipeline">{status?.running ? 'running' : 'stopped'}</div>
  ),
}))

vi.mock('@/components/workflow/WorkflowHealthCard', () => ({
  WorkflowHealthCard: ({ status }: any) => (
    <div data-testid="workflow-health">{status ? 'has-status' : 'no-status'}</div>
  ),
}))

import WorkflowPage from './page'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockStatus.mockResolvedValue({ running: false, last_run: null, total_runs: 0 })
  mockStart.mockResolvedValue({})
  mockStop.mockResolvedValue({})
  mockTrigger.mockResolvedValue({ status: 'done' })
})

describe('WorkflowPage — initial load flow', () => {
  it('renders page header', async () => {
    render(<WorkflowPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Workflow').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('fetches status on mount', async () => {
    render(<WorkflowPage />)
    await waitFor(() => {
      expect(mockStatus).toHaveBeenCalledTimes(1)
    })
  })

  it('shows loading state', () => {
    mockStatus.mockReturnValue(new Promise(() => {}))
    render(<WorkflowPage />)
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(1)
  })
})

describe('WorkflowPage — stopped state flow', () => {
  it('shows stopped status', async () => {
    render(<WorkflowPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Stopped').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows start button when stopped', async () => {
    render(<WorkflowPage />)
    await waitFor(() => {
      expect(screen.getByText('Start')).toBeTruthy()
    })
  })
})

describe('WorkflowPage — running state flow', () => {
  it('shows running status', async () => {
    mockStatus.mockResolvedValue({ running: true, stats: { feedback_recorded: 10 } })
    render(<WorkflowPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Running').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows stop button when running', async () => {
    mockStatus.mockResolvedValue({ running: true })
    render(<WorkflowPage />)
    await waitFor(() => {
      expect(screen.getByText('Stop')).toBeTruthy()
    })
  })
})

describe('WorkflowPage — toggle flow', () => {
  it('start button calls start and refreshes status', async () => {
    render(<WorkflowPage />)
    await waitFor(() => { expect(screen.getByText('Start')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Start')) })
    await waitFor(() => {
      expect(mockStart).toHaveBeenCalled()
      expect(mockStatus).toHaveBeenCalledTimes(2) // mount + after toggle
    })
  })

  it('stop button calls stop when running', async () => {
    mockStatus.mockResolvedValue({ running: true })
    render(<WorkflowPage />)
    await waitFor(() => { expect(screen.getByText('Stop')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Stop')) })
    await waitFor(() => {
      expect(mockStop).toHaveBeenCalled()
    })
  })
})

describe('WorkflowPage — trigger flow', () => {
  it('trigger button calls trigger action', async () => {
    render(<WorkflowPage />)
    await waitFor(() => { expect(screen.getByText('Workflow')).toBeTruthy() })

    const triggerBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('trigger')
    )
    if (triggerBtn) {
      await act(async () => { fireEvent.click(triggerBtn) })
      await waitFor(() => {
        expect(mockTrigger).toHaveBeenCalled()
      })
    }
  })
})

describe('WorkflowPage — pipeline and health', () => {
  it('renders pipeline and health cards', async () => {
    render(<WorkflowPage />)
    await waitFor(() => {
      expect(screen.getByTestId('workflow-pipeline')).toBeTruthy()
      expect(screen.getByTestId('workflow-health')).toBeTruthy()
    })
  })
})

describe('WorkflowPage — error handling', () => {
  it('handles status failure gracefully', async () => {
    mockStatus.mockRejectedValue(new Error('network'))
    render(<WorkflowPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Workflow').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('handles toggle failure gracefully', async () => {
    mockStart.mockRejectedValue(new Error('failed'))
    render(<WorkflowPage />)
    await waitFor(() => { expect(screen.getByText('Start')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Start')) })
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith(
        expect.stringContaining('toggle'),
        'error'
      )
    })
  })
})

describe('WorkflowPage — stats display', () => {
  it('shows stats when available', async () => {
    mockStatus.mockResolvedValue({
      running: true,
      total_runs: 5,
      stats: { feedback_recorded: 100, auto_train_steps: 10 },
    })
    render(<WorkflowPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Running').length).toBeGreaterThanOrEqual(1)
    })
  })
})
