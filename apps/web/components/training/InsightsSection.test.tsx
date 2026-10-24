import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    Card: passthrough,
    CardContent: passthrough,
    CardHeader: passthrough,
    CardTitle: ({ children }: any) => <h3>{children}</h3>,
    CardDescription: ({ children }: any) => <p>{children}</p>,
    Button: ({ children, onClick, disabled }: any) => <button onClick={onClick} disabled={disabled}>{children}</button>,
    Badge: ({ children }: any) => <span>{children}</span>,
    Skeleton: () => <div data-testid="skeleton" />,
    IconDownload: () => <span />,
    IconRefresh: () => <span />,
  }
})

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
  ComposedChart: ({ children }: any) => <div>{children}</div>,
  BarChart: ({ children }: any) => <div>{children}</div>,
  Bar: () => <div />,
  Line: () => <div />,
  Area: () => <div />,
  XAxis: () => <div />,
  YAxis: () => <div />,
  CartesianGrid: () => <div />,
  Tooltip: () => <div />,
  Legend: () => <div />,
}))

vi.mock('lucide-react', () => ({
  Brain: () => <span />,
  Lightbulb: () => <span />,
  Target: () => <span />,
  Sparkles: () => <span />,
  RefreshCw: () => <span />,
  TrendingUp: () => <span />,
  BarChart3: () => <span />,
  Award: () => <span />,
  Tag: () => <span />,
  Download: () => <span />,
  TrendingDown: () => <span />,
  Minus: () => <span />,
}))

const { mockInsights, mockTrends, mockAnalytics } = vi.hoisted(() => ({
  mockInsights: { insights: null as any, trendPoints: [] as any[], loading: false, useConfigStatus: 'idle' as const, loadData: vi.fn(), applyRecommendedConfig: vi.fn() },
  mockTrends: { runs: [] as any[], models: [] as any[], summary: null as any, chartPoints: [] as any[], loading: false, modelFilter: '', datasetFilter: '', setModelFilter: vi.fn(), setDatasetFilter: vi.fn(), loadTrends: vi.fn() },
  mockAnalytics: { analytics: null as any, loading: false, error: null as any, load: vi.fn() },
}))

vi.mock('@/hooks/useTrainingInsights', () => ({ useTrainingInsights: () => mockInsights }))
vi.mock('@/hooks/useTrainingTrends', () => ({ useTrainingTrends: () => mockTrends }))
vi.mock('@/hooks/useTrainingAnalytics', () => ({ useTrainingAnalytics: () => mockAnalytics }))
vi.mock('@/lib/download-utils', () => ({ downloadJson: vi.fn() }))

import { InsightsSection } from './InsightsSection'

afterEach(cleanup)
beforeEach(() => {
  vi.clearAllMocks()
  mockInsights.insights = null
  mockInsights.trendPoints = []
  mockInsights.loading = false
  mockTrends.runs = []
  mockTrends.models = []
  mockTrends.summary = null
  mockTrends.chartPoints = []
  mockTrends.loading = false
  mockAnalytics.analytics = null
  mockAnalytics.loading = false
  mockAnalytics.error = null
})

describe('InsightsSection', () => {
  it('shows loading skeletons while any source loads', () => {
    mockInsights.loading = true
    render(<InsightsSection />)
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(0)
    mockInsights.loading = false
  })

  it('shows the empty state when insights carries a message and no data', () => {
    mockInsights.insights = { total_runs: 0, message: 'Train something first' } as any
    render(<InsightsSection />)
    expect(screen.getByText('No Training History Yet')).toBeTruthy()
    expect(screen.getByText('Train something first')).toBeTruthy()
  })

  it('renders summary, charts, recommendation, and tables when data exists', async () => {
    mockInsights.insights = {
      total_runs: 5, avg_quality: 0.8, avg_loss: 2.1, best_quality: 0.9,
      best_config: { model: 'gpt2', epochs: 3 }, trend: 'improving', recommendation: 'Try more epochs',
    } as any
    mockTrends.summary = { total_runs: 5, avg_quality: 0.8, best_quality: 0.9, avg_loss: 2.1, trend: 'improving' }
    mockTrends.chartPoints = [{ name: 'gpt2#abc123', date: 'Jan 1', quality: 0.8, loss: 2.1, perplexity: null, dataset: 'd', model: 'gpt2', method: 'finetune' }]
    mockTrends.models = [{ model: 'gpt2', total_runs: 5, avg_quality: 0.8, best_quality: 0.9, avg_loss: 2.1, latest_quality: 0.9, improving: true }]
    mockTrends.runs = [{ run_id: 'run-abc123xyz', timestamp: 1767225600, model: 'gpt2', dataset: 'd', method: 'finetune', quality_score: 0.8, final_loss: 2.1, perplexity: 8, converged: true, early_stopped: false }]
    mockAnalytics.analytics = {
      total_runs: 5, avg_quality: 0.8, convergence_rate: 0.8, quality_trend: [{ run_id: 'r', quality: 0.8, timestamp: 't' }],
      method_distribution: { finetune: 5 }, model_distribution: { gpt2: 5 }, tag_cloud: { good: 2 },
      best_run: { run_id: 'r1', model: 'gpt2', quality_score: 0.9, method: 'finetune' },
      recent_runs: [],
    } as any
    render(<InsightsSection />)
    expect(screen.getByText('Quality Over Time')).toBeTruthy()
    expect(screen.getByText('Loss Over Time')).toBeTruthy()
    expect(screen.getByText('Best Configuration Found')).toBeTruthy()
    expect(screen.getByText('Try more epochs')).toBeTruthy()
    expect(screen.getByText('Model Breakdown')).toBeTruthy()
    expect(screen.getByText('Run History')).toBeTruthy()
    expect(screen.getByText('Best Run')).toBeTruthy()
    await act(async () => { fireEvent.click(screen.getByText('Use recommended config')) })
    expect(mockInsights.applyRecommendedConfig).toHaveBeenCalled()
  })

  it('refresh button reloads all three sources', async () => {
    render(<InsightsSection />)
    await act(async () => { fireEvent.click(screen.getByText('Refresh')) })
    await waitFor(() => {
      expect(mockInsights.loadData).toHaveBeenCalled()
      expect(mockTrends.loadTrends).toHaveBeenCalled()
      expect(mockAnalytics.load).toHaveBeenCalled()
    })
  })
})
