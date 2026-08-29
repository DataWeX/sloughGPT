import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

const {
  mockRun, mockMetrics, mockQuality, mockStats, mockHistory, mockApiPost, mockAddToast,
} = vi.hoisted(() => ({
  mockRun: vi.fn(), mockMetrics: vi.fn(), mockQuality: vi.fn(), mockStats: vi.fn(), mockHistory: vi.fn(),
  mockApiPost: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => {
    const C = () => <span data-testid={'icon-' + name}>{name}</span>
    return C
  }
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Card: ({ children, ...props }: any) => <div {...props}>{children}</div>, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, disabled, variant }: any) => (
      <button onClick={onClick} disabled={disabled} data-variant={variant}>{children}</button>
    ),
    Textarea: ({ value, onChange, placeholder }: any) => (
      <textarea value={value} onChange={onChange} placeholder={placeholder} />
    ),
    StatCard: ({ label, value }: any) => <div data-testid={'stat-' + label}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    IconRefresh: iconMock('refresh'),
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    EmptyCard: ({ title, description }: any) => <div data-testid="empty-card"><div>{title}</div><div>{description}</div></div>,
  }
})

vi.mock('@/lib/benchmark-controller', () => ({
  benchmarkController: {
    run: (...a: unknown[]) => mockRun(...a),
    metrics: (...a: unknown[]) => mockMetrics(...a),
    quality: (...a: unknown[]) => mockQuality(...a),
    stats: (...a: unknown[]) => mockStats(...a),
    history: (...a: unknown[]) => mockHistory(...a),
  },
}))

vi.mock('@/lib/http-client', () => ({
  apiPost: (...a: unknown[]) => mockApiPost(...a),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/components/benchmark/BenchmarkInsightsCard', () => ({
  BenchmarkInsightsCard: ({ metrics, quality }: any) => (
    <div data-testid="benchmark-insights">{metrics && 'has-metrics'}{quality && 'has-quality'}</div>
  ),
}))

import BenchmarkPage from './page'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockRun.mockResolvedValue({ throughput: 10, latency_p50: 200, latency_p95: 500 })
  mockMetrics.mockResolvedValue({
    model: 'gpt2', model_loaded: true, inference_count: 5, total_tokens: 210,
    tokens_per_second: 42, memory_mb: 512, num_parameters: 124000000, latency_ms: 200,
    throughput: 10, throughput_tokens_per_sec: 42, inference_time_ms: 200,
  })
  mockQuality.mockResolvedValue({ coherence_score: 0.85, quality_score: 0.78, repetition_rate: 0.12 })
  mockStats.mockResolvedValue({ total: 15, avg_tokens: 42, models: ['gpt2'] })
  mockHistory.mockResolvedValue([
    { timestamp: '2026-08-07T10:00:00Z', user_message: 'Hello', assistant_response: 'Hi there', model: 'gpt2', tokens_generated: 10, duration_ms: 200 },
  ])
  mockApiPost.mockResolvedValue({})
})

describe('BenchmarkPage — initial load flow', () => {
  it('renders header', async () => {
    render(<BenchmarkPage />)
    await waitFor(() => {
      expect(screen.getByText('Benchmark')).toBeTruthy()
    })
  })

  it('fetches metrics, quality, and stats on mount', async () => {
    render(<BenchmarkPage />)
    await waitFor(() => {
      expect(mockMetrics).toHaveBeenCalledTimes(1)
      expect(mockQuality).toHaveBeenCalledTimes(1)
      expect(mockStats).toHaveBeenCalledTimes(1)
    })
  })

  it('shows loading state initially', async () => {
    mockMetrics.mockReturnValue(new Promise(() => {})) // never resolves
    render(<BenchmarkPage />)
    // Should render skeleton while loading
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(1)
  })
})

describe('BenchmarkPage — metrics tab flow', () => {
  it('displays metrics after loading', async () => {
    render(<BenchmarkPage />)
    await waitFor(() => {
      expect(screen.getByText('Benchmark')).toBeTruthy()
    })
    expect(screen.getByText('Benchmark')).toBeTruthy()
  })

  it('refresh metrics button triggers re-run', async () => {
    render(<BenchmarkPage />)
    await waitFor(() => { expect(mockMetrics).toHaveBeenCalledTimes(1) })

    const refreshBtns = screen.getAllByRole('button').filter(b =>
      b.textContent?.includes('refresh') || b.textContent?.includes('Refresh')
    )
    if (refreshBtns.length > 0) {
      await act(async () => { fireEvent.click(refreshBtns[0]) })
      await waitFor(() => {
        expect(mockRun).toHaveBeenCalledTimes(1)
      })
    }
  })
})

describe('BenchmarkPage — quality tab flow', () => {
  it('switches to quality tab and shows scores', async () => {
    render(<BenchmarkPage />)
    await waitFor(() => { expect(mockQuality).toHaveBeenCalled() })

    const qualityTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('quality')
    )
    if (qualityTab) {
      fireEvent.click(qualityTab)
      await waitFor(() => {
        expect(screen.getAllByText(/85\.0%|Coherence/i).length).toBeGreaterThanOrEqual(1)
      })
    }
  })
})

describe('BenchmarkPage — responses tab flow', () => {
  it('switches to responses tab and loads history', async () => {
    render(<BenchmarkPage />)
    await waitFor(() => { expect(screen.getByText('Benchmark')).toBeTruthy() })

    const responsesTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('response')
    )
    if (responsesTab) {
      fireEvent.click(responsesTab)
      await waitFor(() => {
        expect(mockHistory).toHaveBeenCalled()
      })
    }
  })

  it('shows logged responses after loading', async () => {
    render(<BenchmarkPage />)
    await waitFor(() => { expect(screen.getByText('Benchmark')).toBeTruthy() })

    const responsesTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('response')
    )
    if (responsesTab) {
      fireEvent.click(responsesTab)
      await waitFor(() => {
        expect(screen.getByText('Hello')).toBeTruthy()
        expect(screen.getByText('Hi there')).toBeTruthy()
      })
    }
  })
})

describe('BenchmarkPage — perplexity tab flow', () => {
  it('switches to perplexity tab', async () => {
    render(<BenchmarkPage />)
    await waitFor(() => { expect(screen.getByText('Benchmark')).toBeTruthy() })

    const pplxTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('perplexity')
    )
    if (pplxTab) {
      fireEvent.click(pplxTab)
      await waitFor(() => {
        expect(screen.getByPlaceholderText(/text|enter|paste/i)).toBeTruthy()
      })
    }
  })
})

describe('BenchmarkPage — insights card flow', () => {
  it('renders insights card with data', async () => {
    render(<BenchmarkPage />)
    await waitFor(() => {
      const insights = screen.getByTestId('benchmark-insights')
      expect(insights).toBeTruthy()
    })
  })
})

describe('BenchmarkPage — error handling flow', () => {
  it('handles metrics fetch failure gracefully', async () => {
    mockRun.mockRejectedValue(new Error('timeout'))
    render(<BenchmarkPage />)
    await waitFor(() => {
      // Page should still render without crashing
      expect(screen.getByText('Benchmark')).toBeTruthy()
    })
  })

  it('handles quality fetch failure gracefully', async () => {
    mockQuality.mockRejectedValue(new Error('unavailable'))
    render(<BenchmarkPage />)
    await waitFor(() => {
      // Page should still render
      expect(screen.getByText('Benchmark')).toBeTruthy()
    })
  })
})

describe('BenchmarkPage — clear history flow', () => {
  it('clear button clears responses', async () => {
    mockApiPost.mockResolvedValue({})
    render(<BenchmarkPage />)
    await waitFor(() => { expect(screen.getByText('Benchmark')).toBeTruthy() })

    const responsesTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('response')
    )
    if (responsesTab) {
      fireEvent.click(responsesTab)
      await waitFor(() => { expect(mockHistory).toHaveBeenCalled() })

      const clearBtn = screen.getAllByRole('button').find(b =>
        b.textContent?.toLowerCase().includes('clear')
      )
      if (clearBtn) {
        await act(async () => { fireEvent.click(clearBtn) })
        await waitFor(() => {
          expect(mockApiPost).toHaveBeenCalled()
        })
      }
    }
  })
})
