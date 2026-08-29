// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { TrainingAnalyticsCard } from './TrainingAnalyticsCard'
import { trainingJobsController } from '@/lib/training-controller'

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    list: vi.fn(),
  },
}))

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  PieChart: ({ children }: any) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => null,
  Cell: () => null,
  BarChart: ({ children }: any) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => null,
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  CartesianGrid: () => null,
  Legend: () => null,
}))

describe('TrainingAnalyticsCard', () => {
  const mockList = vi.mocked(trainingJobsController.list)
  const mockToast = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows loading state', () => {
    mockList.mockImplementation(() => new Promise(() => {}))
    render(<TrainingAnalyticsCard addToast={mockToast} />)
    expect(screen.getByText('Loading...')).toBeTruthy()
  })

  it('shows empty state when no jobs', async () => {
    mockList.mockResolvedValue([])
    render(<TrainingAnalyticsCard addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('No training data yet. Complete training to see analytics.')).toBeTruthy()
    })
  })

  it('renders analytics with job data', async () => {
    mockList.mockResolvedValue([
      { id: '1', name: 'Job 1', status: 'completed', progress: 100, created_at: '2026-01-15T10:00:00Z', method: 'distill', loss: 0.5 },
      { id: '2', name: 'Job 2', status: 'failed', progress: 50, created_at: '2026-01-14T10:00:00Z', method: 'native', error: 'OOM' },
    ] as any)
    render(<TrainingAnalyticsCard addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('Total runs')).toBeTruthy()
      expect(screen.getByText('Completed')).toBeTruthy()
      expect(screen.getByText('Failed')).toBeTruthy()
      expect(screen.getByText('Avg loss')).toBeTruthy()
    })
  })

  it('shows summary stats', async () => {
    mockList.mockResolvedValue([
      { id: '1', name: 'Job 1', status: 'completed', progress: 100, created_at: '2026-01-15T10:00:00Z', loss: 0.5 },
      { id: '2', name: 'Job 2', status: 'completed', progress: 100, created_at: '2026-01-14T10:00:00Z', loss: 0.3 },
    ] as any)
    render(<TrainingAnalyticsCard addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('Total runs')).toBeTruthy()
      expect(screen.getByText('Avg loss')).toBeTruthy()
    })
    // Check that the stats are rendered
    expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
  })

  it('shows method distribution', async () => {
    mockList.mockResolvedValue([
      { id: '1', name: 'Job 1', status: 'completed', progress: 100, created_at: '2026-01-15T10:00:00Z', method: 'distill' },
      { id: '2', name: 'Job 2', status: 'completed', progress: 100, created_at: '2026-01-14T10:00:00Z', method: 'native' },
    ] as any)
    const { unmount } = render(<TrainingAnalyticsCard addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Method distribution').length).toBeGreaterThan(0)
    })
    expect(screen.getAllByText('distill').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('native').length).toBeGreaterThanOrEqual(1)
    unmount()
  })

  it('shows error on fetch failure', async () => {
    mockList.mockRejectedValue(new Error('Network error'))
    render(<TrainingAnalyticsCard addToast={mockToast} />)
    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith('Could not fetch training data', 'error')
    })
  })
})
