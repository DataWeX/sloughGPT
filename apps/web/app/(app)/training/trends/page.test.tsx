import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, act } from '@testing-library/react'

// ── strui mock ──
vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Card: passthrough,
    CardContent: ({ children, className }: any) => <div className={className}>{children}</div>,
    CardHeader: ({ children, className }: any) => <div className={className}>{children}</div>,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    CardDescription: ({ children }: any) => <p>{children}</p>,
    Button: ({ children, onClick, variant, size, className, disabled }: any) => (
      <button onClick={onClick} className={className} disabled={disabled} data-variant={variant}>{children}</button>
    ),
    Input: ({ value, onChange, placeholder }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} />
    ),
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
  }
})

// ── recharts mock ──
vi.mock('recharts', () => ({
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  CartesianGrid: () => null,
  Legend: () => null,
  Area: () => null,
  ComposedChart: ({ children }: any) => <div data-testid="composed-chart">{children}</div>,
  Bar: () => null,
  BarChart: ({ children }: any) => <div data-testid="bar-chart">{children}</div>,
}))

// ── lucide-react mock ──
vi.mock('lucide-react', () => {
  const iconMock = (name: string) => {
    const C = ({ className }: any) => <span className={className} data-testid={`icon-${name}`} />
    C.displayName = `Icon${name}`
    return C
  }
  return {
    TrendingUp: iconMock('TrendingUp'),
    TrendingDown: iconMock('TrendingDown'),
    Minus: iconMock('Minus'),
    RefreshCw: iconMock('RefreshCw'),
  }
})

// ── component mocks ──
vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title }: any) => (
    <div data-testid="page-container" data-title={title}>{children}</div>
  ),
}))

vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ left }: any) => <div data-testid="app-route-header">{left}</div>,
  AppRouteHeaderLead: ({ title }: any) => <span>{title}</span>,
}))

// ── apiGet mock ──
const mockApiGet = vi.fn()
vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: any[]) => mockApiGet(...args),
}))

import TrainingTrendsPage from './page'

afterEach(() => { cleanup() })
beforeEach(() => { vi.clearAllMocks() })

const mockTrendData = {
  runs: [
    {
      run_id: 'run-abc123def456',
      timestamp: 1700000000,
      dataset: 'wiki',
      dataset_size: 1000,
      model: 'gpt2',
      method: 'sft',
      epochs: 3,
      batch_size: 16,
      learning_rate: 0.0001,
      final_loss: 0.5,
      best_loss: 0.4,
      perplexity: 8.2,
      converged: true,
      early_stopped: false,
      quality_score: 0.85,
      training_time_s: 120,
    },
    {
      run_id: 'run-xyz789ghi012',
      timestamp: 1700100000,
      dataset: 'code',
      dataset_size: 2000,
      model: 'llama',
      method: 'lora',
      epochs: 5,
      batch_size: 8,
      learning_rate: 0.00005,
      final_loss: 0.3,
      best_loss: 0.25,
      perplexity: 5.1,
      converged: false,
      early_stopped: true,
      quality_score: 0.92,
      training_time_s: 300,
    },
  ],
  models: [
    {
      model: 'gpt2',
      total_runs: 1,
      avg_quality: 0.85,
      best_quality: 0.85,
      avg_loss: 0.5,
      latest_quality: 0.85,
      improving: false,
    },
    {
      model: 'llama',
      total_runs: 1,
      avg_quality: 0.92,
      best_quality: 0.92,
      avg_loss: 0.3,
      latest_quality: 0.92,
      improving: true,
    },
  ],
  summary: {
    total_runs: 2,
    avg_quality: 0.885,
    best_quality: 0.92,
    avg_loss: 0.4,
    trend: 'improving' as const,
  },
}

describe('TrainingTrendsPage', () => {
  it('renders without crashing', async () => {
    mockApiGet.mockResolvedValue(mockTrendData)
    render(<TrainingTrendsPage />)
    expect(document.body).toBeTruthy()
    await act(async () => {})
  })

  it('renders page title', async () => {
    mockApiGet.mockResolvedValue(mockTrendData)
    render(<TrainingTrendsPage />)
    expect(screen.getByText('Training Trends')).toBeTruthy()
    await act(async () => {})
  })

  it('shows loading state with skeletons', async () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<TrainingTrendsPage />)
    const skeletons = screen.getAllByTestId('skeleton')
    expect(skeletons.length).toBeGreaterThanOrEqual(2)
    await act(async () => {})
  })

  it('shows summary cards after data loads', async () => {
    mockApiGet.mockResolvedValue(mockTrendData)
    render(<TrainingTrendsPage />)
    await waitFor(() => {
      expect(screen.getByText('Total Runs')).toBeTruthy()
      expect(screen.getByText('Avg Quality')).toBeTruthy()
      expect(screen.getByText('Best Quality')).toBeTruthy()
      expect(screen.getByText('Trend')).toBeTruthy()
    })
    expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('improving')).toBeTruthy()
  })

  it('displays chart cards after data loads', async () => {
    mockApiGet.mockResolvedValue(mockTrendData)
    render(<TrainingTrendsPage />)
    await waitFor(() => {
      expect(screen.getByText('Quality Over Time')).toBeTruthy()
      expect(screen.getByText('Loss Over Time')).toBeTruthy()
      expect(screen.getByText('Model Breakdown')).toBeTruthy()
      expect(screen.getByText('Run History')).toBeTruthy()
    })
  })

  it('shows model breakdown table', async () => {
    mockApiGet.mockResolvedValue(mockTrendData)
    render(<TrainingTrendsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('gpt2').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('llama').length).toBeGreaterThanOrEqual(1)
      expect(screen.getByText('↑ improving')).toBeTruthy()
      expect(screen.getByText('— stable')).toBeTruthy()
    })
  })

  it('shows run history table', async () => {
    mockApiGet.mockResolvedValue(mockTrendData)
    render(<TrainingTrendsPage />)
    await waitFor(() => {
      expect(screen.getByText('Run History')).toBeTruthy()
      expect(screen.getByText('converged')).toBeTruthy()
      expect(screen.getByText('early stopped')).toBeTruthy()
    })
  })

  it('shows empty state when no runs exist', async () => {
    mockApiGet.mockResolvedValue({ runs: [], models: [], summary: { total_runs: 0, avg_quality: 0, best_quality: 0, avg_loss: 0, trend: 'stable' } })
    render(<TrainingTrendsPage />)
    await waitFor(() => {
      expect(screen.getByText('No training runs recorded yet.')).toBeTruthy()
    })
  })

  it('handles error state gracefully', async () => {
    mockApiGet.mockRejectedValue(new Error('Network error'))
    render(<TrainingTrendsPage />)
    await waitFor(() => {
      expect(screen.getByText('No training runs recorded yet.')).toBeTruthy()
    })
    await act(async () => {})
  })

  it('calls apiGet on mount', async () => {
    mockApiGet.mockResolvedValue(mockTrendData)
    render(<TrainingTrendsPage />)
    expect(mockApiGet).toHaveBeenCalledWith('/training/trends')
    await act(async () => {})
  })

  it('refresh button re-fetches data', async () => {
    mockApiGet.mockResolvedValue(mockTrendData)
    render(<TrainingTrendsPage />)
    await waitFor(() => { expect(screen.getByText('Refresh')).toBeTruthy() })
    expect(mockApiGet).toHaveBeenCalledTimes(1)
    await act(async () => { screen.getByText('Refresh').click() })
    await waitFor(() => { expect(mockApiGet).toHaveBeenCalledTimes(2) })
  })

  it('passes filter params to apiGet', async () => {
    mockApiGet.mockResolvedValue(mockTrendData)
    render(<TrainingTrendsPage />)
    await waitFor(() => { expect(screen.getByPlaceholderText('Filter by model...')).toBeTruthy() })
    await act(async () => {
      const input = screen.getByPlaceholderText('Filter by model...')
      const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!
      nativeInputValueSetter.call(input, 'gpt2')
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await act(async () => { screen.getByText('Refresh').click() })
    await waitFor(() => { expect(mockApiGet).toHaveBeenCalledWith('/training/trends?model=gpt2') })
  })

  it('displays empty state when apiGet returns empty runs', async () => {
    mockApiGet.mockResolvedValue({ runs: [], models: [], summary: { total_runs: 0, avg_quality: 0, best_quality: 0, avg_loss: 0, trend: 'stable' } })
    render(<TrainingTrendsPage />)
    await waitFor(() => {
      expect(screen.getByText('No training runs recorded yet.')).toBeTruthy()
      expect(screen.getByText('Complete a training run to see trends here.')).toBeTruthy()
    })
  })
})
