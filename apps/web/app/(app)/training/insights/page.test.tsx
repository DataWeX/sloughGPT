import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({
  usePathname: () => '/training/insights',
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), back: vi.fn(), prefetch: vi.fn() }),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled, variant }: any) => (
      <button onClick={onClick} disabled={disabled} data-variant={variant}>{children}</button>
    ),
    Card: passthrough,
    CardContent: passthrough,
    CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    CardDescription: ({ children }: any) => <p>{children}</p>,
    Badge: ({ children, variant }: any) => <span data-variant={variant}>{children}</span>,
    Skeleton: () => <div data-testid="skeleton" />,
  }
})

vi.mock('recharts', () => {
  const p = ({ children }: any) => <div>{children}</div>
  return {
    ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
    ComposedChart: ({ children }: any) => <div data-testid="composed-chart">{children}</div>,
    LineChart: p, Line: p,
    XAxis: () => null,
    YAxis: () => null,
    Tooltip: () => null,
    Legend: () => null,
    CartesianGrid: () => null,
    Area: () => null,
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title }: any) => <div data-testid="page-container" data-title={title}>{children}</div>,
}))

vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ left }: any) => <div data-testid="app-route-header">{left}</div>,
  AppRouteHeaderLead: ({ title }: any) => <span>{title}</span>,
}))

const mockGetAdaptiveInsights = vi.fn()
const mockUpdateTraining = vi.fn()
vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    getAdaptiveInsights: (...args: any[]) => mockGetAdaptiveInsights(...args),
    updateTraining: (...args: any[]) => mockUpdateTraining(...args),
  },
}))

const mockApiGet = vi.fn()
vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: any[]) => mockApiGet(...args),
}))

import TrainingInsightsPage from './page'

describe('TrainingInsightsPage', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders without crashing', async () => {
    mockGetAdaptiveInsights.mockResolvedValue({
      total_runs: 3,
      avg_quality: 0.8,
      best_quality: 0.9,
      avg_loss: 0.5,
      trend: 'improving',
      recommendation: 'Try more epochs',
      best_config: { model: 'gpt2', lr: 0.001 },
    })
    mockApiGet.mockResolvedValue({
      runs: [],
      summary: { total_runs: 3, avg_quality: 0.8, best_quality: 0.9, avg_loss: 0.5, trend: 'improving' },
    })

    render(<TrainingInsightsPage />)
    await waitFor(() => {
      expect(screen.getByText('Adaptive Insights')).toBeTruthy()
    })
  })

  it('renders the page title via PageContainer', async () => {
    mockGetAdaptiveInsights.mockResolvedValue({
      total_runs: 3,
      avg_quality: 0.8,
      best_quality: 0.9,
      avg_loss: 0.5,
      trend: 'improving',
      recommendation: 'Try more epochs',
      best_config: { model: 'gpt2', lr: 0.001 },
    })
    mockApiGet.mockResolvedValue({
      runs: [],
      summary: { total_runs: 3, avg_quality: 0.8, best_quality: 0.9, avg_loss: 0.5, trend: 'improving' },
    })

    render(<TrainingInsightsPage />)
    await waitFor(() => {
      const container = screen.getByTestId('page-container')
      expect(container.getAttribute('data-title')).toBe('Adaptive Insights')
    })
  })

  it('shows loading skeleton while data is being fetched', async () => {
    let resolveInsights: any
    mockGetAdaptiveInsights.mockImplementation(() => new Promise(r => { resolveInsights = r }))
    mockApiGet.mockResolvedValue({ runs: [], summary: {} })

    render(<TrainingInsightsPage />)

    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(0)

    await act(async () => {
      resolveInsights({
        total_runs: 0,
        avg_quality: 0,
        best_quality: 0,
        avg_loss: 0,
        trend: 'stable',
        recommendation: '',
        best_config: {},
      })
    })
  })

  it('shows no-training-history message when insights has a message and no data', async () => {
    mockGetAdaptiveInsights.mockResolvedValue({
      total_runs: 0,
      avg_quality: 0,
      best_quality: 0,
      avg_loss: 0,
      trend: 'stable',
      recommendation: '',
      best_config: {},
      message: 'No training data available yet.',
    })
    mockApiGet.mockResolvedValue({ runs: [], summary: {} })

    render(<TrainingInsightsPage />)
    await waitFor(() => {
      expect(screen.getByText('No Training History Yet')).toBeTruthy()
      expect(screen.getByText('No training data available yet.')).toBeTruthy()
    })
  })

  it('renders summary cards after data loads', async () => {
    mockGetAdaptiveInsights.mockResolvedValue({
      total_runs: 5,
      avg_quality: 0.85,
      best_quality: 0.92,
      avg_loss: 0.42,
      trend: 'improving',
      recommendation: 'Keep going!',
      best_config: { model: 'gpt2', lr: 0.001 },
    })
    mockApiGet.mockResolvedValue({
      runs: [],
      summary: { total_runs: 5, avg_quality: 0.85, best_quality: 0.92, avg_loss: 0.42, trend: 'improving' },
    })

    render(<TrainingInsightsPage />)
    await waitFor(() => {
      expect(screen.getByText('Total Runs')).toBeTruthy()
      expect(screen.getByText('Avg Quality')).toBeTruthy()
      expect(screen.getByText('Best Quality')).toBeTruthy()
      expect(screen.getByText('Trend')).toBeTruthy()
    })
  })

  it('displays the correct summary values', async () => {
    mockGetAdaptiveInsights.mockResolvedValue({
      total_runs: 7,
      avg_quality: 0.888,
      best_quality: 0.951,
      avg_loss: 0.333,
      trend: 'improving',
      recommendation: 'Great progress!',
      best_config: { model: 'gpt2' },
    })
    mockApiGet.mockResolvedValue({
      runs: [],
      summary: { total_runs: 7 },
    })

    render(<TrainingInsightsPage />)
    await waitFor(() => {
      expect(screen.getByText('7')).toBeTruthy()
      expect(screen.getByText('0.888')).toBeTruthy()
      expect(screen.getAllByText('0.951').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('improving', { exact: false }).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders best configuration card', async () => {
    mockGetAdaptiveInsights.mockResolvedValue({
      total_runs: 3,
      avg_quality: 0.8,
      best_quality: 0.9,
      avg_loss: 0.5,
      trend: 'improving',
      recommendation: '',
      best_config: { model: 'gpt2', learning_rate: 0.001 },
    })
    mockApiGet.mockResolvedValue({ runs: [], summary: {} })

    render(<TrainingInsightsPage />)
    await waitFor(() => {
      expect(screen.getByText('Best Configuration Found')).toBeTruthy()
      expect(screen.getByText('learning rate')).toBeTruthy()
      expect(screen.getByText('0.001')).toBeTruthy()
    })
  })

  it('renders recommendation card with button', async () => {
    mockGetAdaptiveInsights.mockResolvedValue({
      total_runs: 3,
      avg_quality: 0.8,
      best_quality: 0.9,
      avg_loss: 0.5,
      trend: 'improving',
      recommendation: 'Try increasing batch size.',
      best_config: { model: 'gpt2' },
    })
    mockApiGet.mockResolvedValue({ runs: [], summary: {} })

    render(<TrainingInsightsPage />)
    await waitFor(() => {
      expect(screen.getByText('Next Recommended Config')).toBeTruthy()
      expect(screen.getByText('Try increasing batch size.')).toBeTruthy()
      expect(screen.getByText('Use Recommended Config')).toBeTruthy()
    })
  })

  it('applies recommended config when button is clicked', async () => {
    mockGetAdaptiveInsights.mockResolvedValue({
      total_runs: 3,
      avg_quality: 0.8,
      best_quality: 0.9,
      avg_loss: 0.5,
      trend: 'improving',
      recommendation: 'Try new settings.',
      best_config: { model: 'gpt2' },
    })
    mockApiGet.mockResolvedValue({ runs: [], summary: {} })
    mockUpdateTraining.mockResolvedValue({})

    render(<TrainingInsightsPage />)
    await waitFor(() => {
      expect(screen.getByText('Use Recommended Config')).toBeTruthy()
    })

    await act(async () => {
      screen.getByText('Use Recommended Config').click()
    })

    expect(mockUpdateTraining).toHaveBeenCalledWith(expect.objectContaining({
      preferred_model: 'gpt2',
    }))
    await waitFor(() => {
      expect(screen.getByText('Applied to Training Settings')).toBeTruthy()
    })
  })

  it('renders the "What the Engine Learned" section when data exists', async () => {
    mockGetAdaptiveInsights.mockResolvedValue({
      total_runs: 6,
      avg_quality: 0.8,
      best_quality: 0.9,
      avg_loss: 0.5,
      trend: 'improving',
      recommendation: '',
      best_config: { model: 'gpt2' },
    })
    mockApiGet.mockResolvedValue({ runs: [], summary: {} })

    render(<TrainingInsightsPage />)
    await waitFor(() => {
      expect(screen.getByText('What the Engine Learned')).toBeTruthy()
      expect(screen.getByText('Data Efficiency')).toBeTruthy()
      expect(screen.getByText('Quality Trajectory')).toBeTruthy()
    })
  })

  it('renders refresh button', async () => {
    mockGetAdaptiveInsights.mockResolvedValue({
      total_runs: 1,
      avg_quality: 0.5,
      best_quality: 0.5,
      avg_loss: 1.0,
      trend: 'stable',
      recommendation: '',
      best_config: {},
    })
    mockApiGet.mockResolvedValue({ runs: [], summary: {} })

    render(<TrainingInsightsPage />)
    await waitFor(() => {
      expect(screen.getByText('Refresh')).toBeTruthy()
    })
  })

  it('fetches insights and trend data on mount', async () => {
    mockGetAdaptiveInsights.mockResolvedValue({
      total_runs: 1,
      avg_quality: 0.5,
      best_quality: 0.5,
      avg_loss: 1.0,
      trend: 'stable',
      recommendation: '',
      best_config: {},
    })
    mockApiGet.mockResolvedValue({ runs: [], summary: {} })

    render(<TrainingInsightsPage />)
    await waitFor(() => {
      expect(mockGetAdaptiveInsights).toHaveBeenCalled()
      expect(mockApiGet).toHaveBeenCalledWith('/training/trends')
    })
  })

  it('handles error when insights fetch fails gracefully', async () => {
    mockGetAdaptiveInsights.mockRejectedValue(new Error('fail'))
    mockApiGet.mockResolvedValue({ runs: [], summary: {} })

    render(<TrainingInsightsPage />)
    await waitFor(() => {
      expect(screen.getByText('Adaptive Insights')).toBeTruthy()
    })
  })

  it('renders learning curve chart when trend data has runs', async () => {
    mockGetAdaptiveInsights.mockResolvedValue({
      total_runs: 3,
      avg_quality: 0.8,
      best_quality: 0.9,
      avg_loss: 0.5,
      trend: 'improving',
      recommendation: '',
      best_config: { model: 'gpt2' },
    })
    mockApiGet.mockResolvedValue({
      runs: [
        { run_id: '1', timestamp: 1700000000, model: 'gpt2', dataset: 'ds', method: 'finetune', quality_score: 0.7, final_loss: 0.6, learning_rate: 0.001, batch_size: 32, epochs: 3 },
        { run_id: '2', timestamp: 1700100000, model: 'gpt2', dataset: 'ds', method: 'finetune', quality_score: 0.85, final_loss: 0.4, learning_rate: 0.001, batch_size: 32, epochs: 3 },
      ],
      summary: { total_runs: 3, avg_quality: 0.8, best_quality: 0.9, avg_loss: 0.5, trend: 'improving' },
    })

    render(<TrainingInsightsPage />)
    await waitFor(() => {
      expect(screen.getByText('Learning Curve')).toBeTruthy()
      expect(screen.getByTestId('responsive-container')).toBeTruthy()
    })
  })
})
