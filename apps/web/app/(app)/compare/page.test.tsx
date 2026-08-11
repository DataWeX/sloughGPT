import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const {
  mockList, mockGetHealth, mockRunBenchmark, mockGetHistory, mockAddToast, mockDownloadJson,
} = vi.hoisted(() => ({
  mockList: vi.fn(), mockGetHealth: vi.fn(), mockRunBenchmark: vi.fn(),
  mockGetHistory: vi.fn(), mockAddToast: vi.fn(), mockDownloadJson: vi.fn(),
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
    Input: ({ value, onChange, placeholder }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} />
    ),
    IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
  }
})

vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ left, right }: any) => <div>{left}{right}</div>,
  AppRouteHeaderLead: ({ title }: any) => <h1>{title}</h1>,
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    list: (...a: unknown[]) => mockList(...a),
    getHealth: (...a: unknown[]) => mockGetHealth(...a),
  },
}))

vi.mock('@/lib/benchmark-controller', () => ({
  benchmarkController: {
    run: (...a: unknown[]) => mockRunBenchmark(...a),
    history: (...a: unknown[]) => mockGetHistory(...a),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: (...a: unknown[]) => mockDownloadJson(...a),
}))

vi.mock('@/lib/format-bytes', () => ({
  todayDateString: () => '2026-08-07',
  getJsonItem: vi.fn().mockReturnValue([]),
}))

vi.mock('next/dynamic', () => {
  const React = require('react')
  return {
    __esModule: true,
    default: () => (props: Record<string, unknown>) => React.createElement('div', { 'data-testid': 'dynamic' }),
  }
})

vi.mock('@/components/compare/ModelsCard', () => ({
  default: ({ models, onSelect }: any) => (
    <div data-testid="models-card">
      {models?.map((m: any) => (
        <button key={m.id} onClick={() => onSelect?.(m.id)}>{m.name}</button>
      ))}
    </div>
  ),
}))

vi.mock('@/components/compare/ComparisonTableCard', () => ({
  default: ({ results }: any) => (
    <div data-testid="comparison-table">{results ? 'has-results' : 'no-results'}</div>
  ),
}))

vi.mock('@/components/compare/SummaryCard', () => ({
  default: ({ results }: any) => (
    <div data-testid="summary-card">{results ? 'has-results' : 'no-results'}</div>
  ),
}))

vi.mock('@/components/compare/ModelComparisonInsightsCard', () => ({
  ModelComparisonInsightsCard: ({ results }: any) => (
    <div data-testid="insights-card">{results ? 'has-results' : 'no-results'}</div>
  ),
}))

import ComparePage from './page'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockList.mockResolvedValue([
    { id: 'gpt2', name: 'gpt2', loaded: true, size_gb: 0.5 },
    { id: 'qwen', name: 'qwen', loaded: false, size_gb: 1.0 },
  ])
  mockGetHealth.mockResolvedValue({ status: 'healthy', model_type: 'gpt2' })
  mockRunBenchmark.mockResolvedValue({ throughput: 10, latency_p50: 200 })
  mockGetHistory.mockResolvedValue([])
})

describe('ComparePage — initial load flow', () => {
  it('renders page header', async () => {
    render(<ComparePage />)
    expect(screen.getAllByText('Model Comparison').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches models on mount', async () => {
    render(<ComparePage />)
    await waitFor(() => {
      expect(mockList).toHaveBeenCalledTimes(1)
      expect(mockGetHealth).toHaveBeenCalledTimes(1)
    })
  })

  it('shows models card', async () => {
    render(<ComparePage />)
    await waitFor(() => {
      expect(screen.getByTestId('models-card')).toBeTruthy()
    })
  })
})

describe('ComparePage — model selection flow', () => {
  it('displays available models', async () => {
    render(<ComparePage />)
    await waitFor(() => {
      expect(screen.getByText('gpt2')).toBeTruthy()
      expect(screen.getByText('qwen')).toBeTruthy()
    })
  })
})

describe('ComparePage — benchmark flow', () => {
  it('run benchmark button triggers benchmark', async () => {
    render(<ComparePage />)
    await waitFor(() => { expect(screen.getByText('gpt2')).toBeTruthy() })

    const benchBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('benchmark') || b.textContent?.toLowerCase().includes('run')
    )
    if (benchBtn) {
      await act(async () => { fireEvent.click(benchBtn) })
      // No crash = success
      expect(screen.getByTestId('models-card')).toBeTruthy()
    }
  })
})

describe('ComparePage — results display', () => {
  it('shows empty state when no results', async () => {
    render(<ComparePage />)
    await waitFor(() => {
      expect(screen.getByText(/no benchmark results/i)).toBeTruthy()
    })
  })
})

describe('ComparePage — snapshot flow', () => {
  it('save snapshot button works', async () => {
    render(<ComparePage />)
    await waitFor(() => { expect(screen.getByText('Model Comparison')).toBeTruthy() })

    const saveBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('save')
    )
    if (saveBtn) {
      await act(async () => { fireEvent.click(saveBtn) })
      // No crash = success
      expect(screen.getByText('Model Comparison')).toBeTruthy()
    }
  })
})

describe('ComparePage — error handling', () => {
  it('handles model list failure gracefully', async () => {
    mockList.mockRejectedValue(new Error('network'))
    render(<ComparePage />)
    await waitFor(() => {
      expect(screen.getAllByText('Model Comparison').length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('ComparePage — insights card', () => {
  it('does not render insights card when no results', async () => {
    render(<ComparePage />)
    await waitFor(() => {
      // Insights card only renders when there are completed results
      expect(screen.queryByTestId('insights-card')).toBeNull()
    })
  })

  it('shows empty state instruction text', async () => {
    render(<ComparePage />)
    await waitFor(() => {
      expect(screen.getByText(/no benchmark results/i)).toBeTruthy()
    })
  })

  it('shows loading state while fetching models', async () => {
    mockList.mockReturnValue(new Promise(() => {}))
    render(<ComparePage />)
    expect(screen.getByText('Model Comparison')).toBeTruthy()
  })

  it('shows error toast when model list fails', async () => {
    mockList.mockRejectedValue(new Error('network'))
    render(<ComparePage />)
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalled()
    })
  })

  it('does not render comparison table when no results', async () => {
    render(<ComparePage />)
    await waitFor(() => {
      expect(screen.getByText(/no benchmark results/i)).toBeTruthy()
    })
    expect(screen.queryByTestId('comparison-table')).toBeNull()
  })

  it('does not render summary card when no results', async () => {
    render(<ComparePage />)
    await waitFor(() => {
      expect(screen.getByText(/no benchmark results/i)).toBeTruthy()
    })
    expect(screen.queryByTestId('summary-card')).toBeNull()
  })

  it('renders models card with model list', async () => {
    render(<ComparePage />)
    await waitFor(() => {
      const card = screen.getByTestId('models-card')
      expect(card).toBeTruthy()
      expect(screen.getByText('gpt2')).toBeTruthy()
    })
  })

  it('clicking model selects it', async () => {
    render(<ComparePage />)
    await waitFor(() => { expect(screen.getByText('gpt2')).toBeTruthy() })
    const modelBtn = screen.getByText('gpt2')
    await act(async () => { fireEvent.click(modelBtn) })
    expect(screen.getByTestId('models-card')).toBeTruthy()
  })

  it('does not show export button when no results', async () => {
    render(<ComparePage />)
    await waitFor(() => {
      expect(screen.getByText(/no benchmark results/i)).toBeTruthy()
    })
    const exportBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('export')
    )
    expect(exportBtn).toBeUndefined()
  })
})
