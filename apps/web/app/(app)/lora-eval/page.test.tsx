import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const {
  mockRunEval, mockGetHistory, mockAggregate, mockAddToast,
} = vi.hoisted(() => ({
  mockRunEval: vi.fn(), mockGetHistory: vi.fn(), mockAggregate: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, disabled, size, variant }: any) => (
      <button onClick={onClick} disabled={disabled} data-size={size} data-variant={variant}>{children}</button>
    ),
    Input: ({ value, onChange, type, className }: any) => (
      <input value={value} onChange={onChange} type={type} className={className} />
    ),
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  }
})

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/lora-eval-controller', () => ({
  loraEvalController: {
    runEval: (...a: unknown[]) => mockRunEval(...a),
    getHistory: (...a: unknown[]) => mockGetHistory(...a),
    aggregate: (...a: unknown[]) => mockAggregate(...a),
  },
}))

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, headerRight }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p><div>{headerRight}</div>{children}</div>
  ),
}))

import LoraEvalPage from './page'

afterEach(cleanup)

beforeEach(() => {
  vi.restoreAllMocks()
  mockGetHistory.mockResolvedValue([])
})

describe('LoraEvalPage', () => {
  it('renders page header', async () => {
    render(<LoraEvalPage />)
    expect(screen.getByText('LoRA Evaluation')).toBeTruthy()
    expect(screen.getByText(/Evaluate adapter quality/)).toBeTruthy()
  })

  it('fetches history on mount', async () => {
    render(<LoraEvalPage />)
    await waitFor(() => {
      expect(mockGetHistory).toHaveBeenCalledWith(50)
    })
  })

  it('shows loading state initially', () => {
    mockGetHistory.mockReturnValue(new Promise(() => {}))
    render(<LoraEvalPage />)
    expect(screen.getByText('Loading...')).toBeTruthy()
  })

  it('shows empty state when no history', async () => {
    render(<LoraEvalPage />)
    await waitFor(() => {
      expect(screen.getByText(/No evaluations yet/)).toBeTruthy()
    })
  })

  it('displays eval history items', async () => {
    mockGetHistory.mockResolvedValue([
      { status: 'passed', elapsed_ms: 120, report: 'All metrics improved', delta: { verdict: 'improved' } },
      { status: 'failed', elapsed_ms: 80 },
    ])
    render(<LoraEvalPage />)
    await waitFor(() => {
      expect(screen.getAllByText('passed').length).toBeGreaterThanOrEqual(1)
      expect(screen.getByText('failed')).toBeTruthy()
      expect(screen.getByText('120ms')).toBeTruthy()
      expect(screen.getByText('All metrics improved')).toBeTruthy()
    })
  })

  it('displays KPI stats', async () => {
    mockGetHistory.mockResolvedValue([
      { status: 'passed', delta: { verdict: 'improved' } },
      { status: 'passed' },
    ])
    render(<LoraEvalPage />)
    await waitFor(() => {
      const totalCard = screen.getByTestId('stat-Total Evals')
      expect(totalCard.textContent).toContain('2')
    })
  })

  it('run eval calls controller with params', async () => {
    mockRunEval.mockResolvedValue({ status: 'passed', elapsed_ms: 100 })
    render(<LoraEvalPage />)
    await waitFor(() => { expect(screen.getByText('Run Eval')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Run Eval')) })
    await waitFor(() => {
      expect(mockRunEval).toHaveBeenCalledWith('data/user_adapters/best_aggregated.npz', 'assistant')
    })
    expect(mockAddToast).toHaveBeenCalledWith('Eval complete: passed', 'success')
  })

  it('run eval shows error toast on failure', async () => {
    mockRunEval.mockRejectedValue(new Error('eval failed'))
    render(<LoraEvalPage />)
    await waitFor(() => { expect(screen.getByText('Run Eval')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Run Eval')) })
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Could not run eval', 'error')
    })
  })

  it('aggregate calls controller', async () => {
    mockAggregate.mockResolvedValue({ status: 'ok' })
    render(<LoraEvalPage />)
    await waitFor(() => { expect(screen.getByText('Aggregate')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Aggregate')) })
    await waitFor(() => {
      expect(mockAggregate).toHaveBeenCalledWith(10, 5)
    })
    expect(mockAddToast).toHaveBeenCalledWith('Aggregated: ok', 'success')
  })

  it('aggregate shows error toast on failure', async () => {
    mockAggregate.mockRejectedValue(new Error('agg failed'))
    render(<LoraEvalPage />)
    await waitFor(() => { expect(screen.getByText('Aggregate')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Aggregate')) })
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Could not aggregate adapters', 'error')
    })
  })

  it('shows Running... while eval is in progress', async () => {
    let resolveEval: (v: any) => void
    mockRunEval.mockReturnValue(new Promise(r => { resolveEval = r }))

    render(<LoraEvalPage />)
    await waitFor(() => { expect(screen.getByText('Run Eval')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Run Eval')) })
    expect(screen.getByText('Running...')).toBeTruthy()

    await act(async () => { resolveEval!({ status: 'done' }) })
  })

  it('shows Aggregating... while aggregate is in progress', async () => {
    let resolveAgg: (v: any) => void
    mockAggregate.mockReturnValue(new Promise(r => { resolveAgg = r }))

    render(<LoraEvalPage />)
    await waitFor(() => { expect(screen.getByText('Aggregate')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Aggregate')) })
    expect(screen.getByText('Aggregating...')).toBeTruthy()

    await act(async () => { resolveAgg!({ status: 'ok' }) })
  })

  it('refresh button calls fetchHistory', async () => {
    render(<LoraEvalPage />)
    await waitFor(() => { expect(mockGetHistory).toHaveBeenCalledTimes(1) })

    const refreshBtn = screen.getAllByRole('button').find(b => b.textContent === 'Refresh')
    await act(async () => { fireEvent.click(refreshBtn!) })
    await waitFor(() => { expect(mockGetHistory).toHaveBeenCalledTimes(2) })
  })

  it('history fetch error shows toast', async () => {
    mockGetHistory.mockRejectedValue(new Error('network'))
    render(<LoraEvalPage />)
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Could not load eval history', 'error')
    })
  })
})
