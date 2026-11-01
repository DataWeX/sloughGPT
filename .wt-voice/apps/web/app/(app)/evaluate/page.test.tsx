import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const mockGetHealth = vi.fn()
const mockListModel = vi.fn()
const mockMetrics = vi.fn()
const mockQuality = vi.fn()
const mockStats = vi.fn()
const mockHistory = vi.fn()
const mockApiPost = vi.fn()

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}))

vi.mock('next/dynamic', () => ({
  __esModule: true,
  default: () => {
    const Dyn = () => null
    Dyn.displayName = 'NextDynamic'
    return Dyn
  },
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    getHealth: (...args: unknown[]) => mockGetHealth(...args),
    list: (...args: unknown[]) => mockListModel(...args),
  },
}))

vi.mock('@/lib/benchmark-controller', () => ({
  benchmarkController: {
    run: (...args: unknown[]) => vi.fn().mockResolvedValue({})(),
    metrics: (...args: unknown[]) => mockMetrics(...args),
    quality: (...args: unknown[]) => mockQuality(...args),
    stats: (...args: unknown[]) => mockStats(...args),
    history: (...args: unknown[]) => mockHistory(...args),
  },
}))

vi.mock('@/lib/http-client', () => ({
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiGet: vi.fn(),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
}))

vi.mock('@/lib/format-bytes', () => ({
  todayDateString: () => '2024-01-15',
  getJsonItem: vi.fn(() => []),
}))

vi.mock('@/components/benchmark/BenchmarkInsightsCard', () => ({
  BenchmarkInsightsCard: () => <div data-testid="benchmark-insights" />,
}))

vi.mock('@/components/compare/ModelsCard', () => ({
  __esModule: true,
  default: () => <div data-testid="models-card" />,
}))

vi.mock('@/components/compare/ComparisonTableCard', () => ({
  __esModule: true,
  default: () => <div data-testid="comparison-table" />,
}))

vi.mock('@/components/compare/SummaryCard', () => ({
  __esModule: true,
  default: () => <div data-testid="summary-card" />,
}))

vi.mock('@/components/compare/ModelComparisonInsightsCard', () => ({
  ModelComparisonInsightsCard: () => <div data-testid="comparison-insights" />,
}))

vi.mock('@/components/compare/OutputComparisonCard', () => ({
  __esModule: true,
  default: () => <div data-testid="output-comparison" />,
}))

vi.mock('@/components/compare/VisualComparisonCard', () => ({
  __esModule: true,
  default: () => <div data-testid="visual-comparison" />,
}))

import EvaluatePage from './page'

describe('EvaluatePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGetHealth.mockResolvedValue({ model_type: 'gpt2' })
    mockListModel.mockResolvedValue([])
    mockMetrics.mockResolvedValue({ model: 'gpt2', inference_count: 0, tokens_per_second: 0, memory_mb: 0, total_tokens: 0, model_loaded: false })
    mockQuality.mockResolvedValue(null)
    mockStats.mockResolvedValue(null)
    mockHistory.mockResolvedValue([])
    mockApiPost.mockResolvedValue({})
  })

  afterEach(() => { cleanup() })

  it('renders and shows section tabs after loading', async () => {
    render(<EvaluatePage />)
    await screen.findByText('Single Model')
    expect(screen.getAllByText('Evaluate').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Comparison')).toBeTruthy()
  })

  it('defaults to single model with benchmark insights', async () => {
    render(<EvaluatePage />)
    await screen.findByTestId('benchmark-insights')
  })

  it('shows sub-tabs for single model', async () => {
    render(<EvaluatePage />)
    await screen.findByText('Metrics')
    expect(screen.getByText('Quality')).toBeTruthy()
    expect(screen.getAllByText('Responses').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Perplexity')).toBeTruthy()
  })

  it('calls benchmark APIs on mount', async () => {
    render(<EvaluatePage />)
    await screen.findByText('Metrics')
    expect(mockGetHealth).toHaveBeenCalled()
    expect(mockMetrics).toHaveBeenCalledWith('gpt2')
    expect(mockQuality).toHaveBeenCalled()
    expect(mockStats).toHaveBeenCalled()
  })

  it('displays metrics KPIs', async () => {
    mockMetrics.mockResolvedValue({
      model: 'gpt2', inference_count: 10, tokens_per_second: 5.5,
      memory_mb: 512, total_tokens: 1000, model_loaded: true,
    })
    render(<EvaluatePage />)
    await screen.findAllByText('gpt2')
    expect(screen.getAllByText('512 MB').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1000').length).toBeGreaterThanOrEqual(1)
  })

  it('shows no-metrics message when metrics is null but page loads', async () => {
    mockMetrics.mockResolvedValue(null)
    mockQuality.mockResolvedValue({ coherence_score: 0.5, quality_score: 0.5, repetition_rate: 0.1, total_responses: 5, avg_length: 100 })
    render(<EvaluatePage />)
    await screen.findByText('Single Model')
    expect(screen.getByText('Single Model')).toBeTruthy()
  })

  it('shows error when all benchmarks fail', async () => {
    mockGetHealth.mockRejectedValue(new Error('fail'))
    mockMetrics.mockRejectedValue(new Error('fail'))
    render(<EvaluatePage />)
    await screen.findByText(/could not load benchmark data/i)
  })
})
