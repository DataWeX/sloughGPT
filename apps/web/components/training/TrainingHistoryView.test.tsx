// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { TrainingHistoryView } from './TrainingHistoryView'
import { trainingJobsController } from '@/lib/training-controller'

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    list: vi.fn(),
  },
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
}))

const JOBS = [
  { id: '1', name: 'Job 1', status: 'completed', progress: 100, created_at: '2026-01-15T10:00:00Z', method: 'distill', loss: 0.5, epochs: 10, checkpoint: 'cp1' },
  { id: '2', name: 'Job 2', status: 'failed', progress: 50, created_at: '2026-01-14T10:00:00Z', method: 'native', error: 'OOM' },
] as any[]

describe('TrainingHistoryView', () => {
  const mockList = vi.mocked(trainingJobsController.list)
  const mockToast = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue(JOBS)
  })

  it('shows loading state', () => {
    mockList.mockImplementation(() => new Promise(() => {}))
    render(<TrainingHistoryView addToast={mockToast} />)
    expect(screen.getByText('Loading...')).toBeTruthy()
  })

  it('shows empty state when no jobs', async () => {
    mockList.mockResolvedValue([])
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('No training jobs yet. Start training to see history.')).toBeTruthy()
    })
  })

  it('renders job list', async () => {
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 1').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Job 2').length).toBeGreaterThan(0)
    })
  })

  it('shows status filter buttons', async () => {
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('All (2)')).toBeTruthy()
    })
  })

  it('expands job details on click', async () => {
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 1').length).toBeGreaterThan(0)
    })
    fireEvent.click(screen.getAllByText('Job 1')[0])
    await waitFor(() => {
      expect(screen.getByText('Loss: 0.5000')).toBeTruthy()
      expect(screen.getByText('Epochs: 10')).toBeTruthy()
      expect(screen.getByText('Checkpoint: cp1')).toBeTruthy()
    })
  })

  it('shows error on fetch failure', async () => {
    mockList.mockRejectedValue(new Error('Network error'))
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith('Could not fetch training history', 'error')
    })
  })

  it('exports training history as JSON', async () => {
    const { downloadJson } = await import('@/lib/download-utils')
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 1').length).toBeGreaterThan(0)
    })
    fireEvent.click(screen.getAllByText('Export')[0])
    await waitFor(() => {
      expect(downloadJson).toHaveBeenCalled()
      expect(mockToast).toHaveBeenCalledWith('Training history exported', 'success')
    })
  })

  it('paginates jobs at 20 per page', async () => {
    const manyJobs = Array.from({ length: 25 }, (_, i) => ({
      id: `${i}`, name: `Job ${i}`, status: 'completed', progress: 100,
      created_at: `2026-01-${String(15 - (i % 15)).padStart(2, '0')}T10:00:00Z`,
    }))
    mockList.mockResolvedValue(manyJobs as any[])
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 0').length).toBeGreaterThan(0)
    })
    expect(screen.queryByText('Job 20')).toBeNull()
    expect(screen.getByText('1–20 of 25')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Next' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    await waitFor(() => {
      expect(screen.getAllByText('Job 20').length).toBeGreaterThan(0)
    })
    expect(screen.queryByText('Job 0')).toBeNull()
    expect(screen.getByText('21–25 of 25')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Prev' }))
    await waitFor(() => {
      expect(screen.getAllByText('Job 0').length).toBeGreaterThan(0)
    })
  })

  it('does not show pagination when 20 or fewer jobs', async () => {
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 1').length).toBeGreaterThan(0)
    })
    expect(screen.queryByText('1–2 of 2')).toBeNull()
  })
})
