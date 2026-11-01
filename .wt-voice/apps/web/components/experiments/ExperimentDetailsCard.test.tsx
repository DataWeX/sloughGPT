// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import React from 'react'
import { ExperimentDetailsCard } from './ExperimentDetailsCard'
import { experimentsController } from '@/lib/experiments-controller'

vi.mock('@/lib/experiments-controller', () => ({
  experimentsController: {
    getExperimentData: vi.fn(),
  },
}))

beforeEach(() => { vi.clearAllMocks() })
afterEach(() => { cleanup() })

const mockData = {
  id: 'exp_1',
  metrics: [
    { metric: 'loss', value: 0.5, step: 10, timestamp: '2026-01-01T00:00:00Z' },
    { metric: 'loss', value: 0.3, step: 20, timestamp: '2026-01-01T00:01:00Z' },
    { metric: 'accuracy', value: 0.85, step: 20, timestamp: '2026-01-01T00:01:00Z' },
  ],
  params: [
    { param: 'learning_rate', value: '0.001', timestamp: '2026-01-01T00:00:00Z' },
    { param: 'batch_size', value: '32', timestamp: '2026-01-01T00:00:00Z' },
  ],
  status: { status: 'completed', completed_at: '2026-01-01T00:05:00Z' },
}

describe('ExperimentDetailsCard', () => {
  it('shows loading state', () => {
    vi.mocked(experimentsController.getExperimentData).mockReturnValue(new Promise(() => {}))
    render(<ExperimentDetailsCard experimentId="exp_1" />)
    expect(screen.getAllByTestId('experiment-details').length).toBeGreaterThanOrEqual(1)
  })

  it('renders metrics and params', async () => {
    vi.mocked(experimentsController.getExperimentData).mockResolvedValue(mockData as any)
    render(<ExperimentDetailsCard experimentId="exp_1" />)
    await waitFor(() => {
      expect(screen.getAllByText('Metrics').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('Parameters').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('loss').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('accuracy').length).toBeGreaterThanOrEqual(1)
  })

  it('shows latest metric values', async () => {
    vi.mocked(experimentsController.getExperimentData).mockResolvedValue(mockData as any)
    render(<ExperimentDetailsCard experimentId="exp_1" />)
    await waitFor(() => {
      expect(screen.getAllByText('0.3000').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('0.8500').length).toBeGreaterThanOrEqual(1)
  })

  it('shows params', async () => {
    vi.mocked(experimentsController.getExperimentData).mockResolvedValue(mockData as any)
    render(<ExperimentDetailsCard experimentId="exp_1" />)
    await waitFor(() => {
      expect(screen.getAllByText('learning_rate').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('0.001').length).toBeGreaterThanOrEqual(1)
  })

  it('shows completed status', async () => {
    vi.mocked(experimentsController.getExperimentData).mockResolvedValue(mockData as any)
    render(<ExperimentDetailsCard experimentId="exp_1" />)
    await waitFor(() => {
      expect(screen.getAllByText('completed').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('returns null when no data', async () => {
    vi.mocked(experimentsController.getExperimentData).mockResolvedValue({
      id: 'exp_1', metrics: [], params: [], status: null,
    } as any)
    const { container } = render(<ExperimentDetailsCard experimentId="exp_1" />)
    await waitFor(() => {
      expect(container.querySelector('[data-testid="experiment-details"]')).toBeNull()
    })
  })

  it('shows empty message when no metrics/params', async () => {
    vi.mocked(experimentsController.getExperimentData).mockResolvedValue({
      id: 'exp_1', metrics: [], params: [], status: { status: 'running' },
    } as any)
    render(<ExperimentDetailsCard experimentId="exp_1" />)
    await waitFor(() => {
      expect(screen.getAllByText('No metrics or parameters logged yet.').length).toBeGreaterThanOrEqual(1)
    })
  })
})
