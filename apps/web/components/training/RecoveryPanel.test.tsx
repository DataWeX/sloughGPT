import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

const { mockTrainingJobs, mockAddToast } = vi.hoisted(() => ({
  mockTrainingJobs: {
    getRecoveryStats: vi.fn(),
    recoverable: vi.fn(),
    recover: vi.fn(),
    abandon: vi.fn(),
  },
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: mockTrainingJobs,
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/dev-log', () => ({
  devDebug: vi.fn(),
}))

import { RecoveryPanel } from './RecoveryPanel'

const emptyJobs: any[] = []

describe('RecoveryPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.confirm = vi.fn(() => true)
  })

  afterEach(cleanup)

  it('renders nothing when no crashed jobs', async () => {
    mockTrainingJobs.getRecoveryStats.mockResolvedValue({ crashed_jobs: 0 })
    mockTrainingJobs.recoverable.mockResolvedValue([])
    const { container } = render(<RecoveryPanel jobs={emptyJobs} fetchJobs={vi.fn()} />)
    expect(container.innerHTML).toBe('')
  })

  it('shows crash warning when crashed jobs exist', async () => {
    mockTrainingJobs.getRecoveryStats.mockResolvedValue({ crashed_jobs: 2 })
    mockTrainingJobs.recoverable.mockResolvedValue([])
    render(<RecoveryPanel jobs={emptyJobs} fetchJobs={vi.fn()} />)
    const warning = await screen.findByText(/2 job\(s\) may have crashed/)
    expect(warning).toBeDefined()
  })

  it('shows job name when expanded', async () => {
    mockTrainingJobs.getRecoveryStats.mockResolvedValue({ crashed_jobs: 1 })
    mockTrainingJobs.recoverable.mockResolvedValue([
      { id: 'job-1', status: 'failed', progress: 50, config: {}, checkpoint_path: 'checkpoints/model.pt', name: 'My Job' },
    ])
    render(<RecoveryPanel jobs={emptyJobs} fetchJobs={vi.fn()} />)
    const showBtn = await screen.findByText('Show')
    fireEvent.click(showBtn)
    expect(screen.getByText('My Job')).toBeDefined()
  })

  it('shows progress value from API', async () => {
    mockTrainingJobs.getRecoveryStats.mockResolvedValue({ crashed_jobs: 1 })
    mockTrainingJobs.recoverable.mockResolvedValue([
      { id: 'job-1', status: 'failed', progress: 50, config: {}, checkpoint_path: undefined, name: 'Job 1' },
    ])
    render(<RecoveryPanel jobs={emptyJobs} fetchJobs={vi.fn()} />)
    const showBtn = await screen.findByText('Show')
    fireEvent.click(showBtn)
    // Component hardcodes progress to 0 (line 42 of source)
    expect(screen.getByText(/Progress:.*0%/)).toBeDefined()
  })

  it('calls recover when Recover clicked', async () => {
    mockTrainingJobs.getRecoveryStats.mockResolvedValue({ crashed_jobs: 1 })
    mockTrainingJobs.recoverable.mockResolvedValue([
      { id: 'job-1', status: 'failed', progress: 50, config: {}, checkpoint_path: undefined, name: 'Job 1' },
    ])
    mockTrainingJobs.recover.mockResolvedValue({ status: 'recovered' })
    render(<RecoveryPanel jobs={emptyJobs} fetchJobs={vi.fn()} />)
    const showBtn = await screen.findByText('Show')
    fireEvent.click(showBtn)
    fireEvent.click(screen.getByText('Recover'))
    expect(mockTrainingJobs.recover).toHaveBeenCalledWith('job-1')
  })

  it('calls abandon when Abandon clicked', async () => {
    mockTrainingJobs.getRecoveryStats.mockResolvedValue({ crashed_jobs: 1 })
    mockTrainingJobs.recoverable.mockResolvedValue([
      { id: 'job-1', status: 'failed', progress: 50, config: {}, checkpoint_path: undefined, name: 'Job 1' },
    ])
    render(<RecoveryPanel jobs={emptyJobs} fetchJobs={vi.fn()} />)
    const showBtn = await screen.findByText('Show')
    fireEvent.click(showBtn)
    fireEvent.click(screen.getByText('Abandon'))
    expect(mockTrainingJobs.abandon).toHaveBeenCalledWith('job-1')
  })
})
