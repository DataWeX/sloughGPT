import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

const mockPush = vi.fn()
vi.mock('@/vite/next-compat/navigation', () => ({ useRouter: () => ({ push: mockPush }) }))

const mockAddToast = vi.fn()
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: any) => selector({ addToast: mockAddToast }),
}))

const mockRecover = vi.fn().mockResolvedValue({
  status: 'recovering',
  recovery_job_id: 'recovery_job-1',
  message: 'Resumed from checkpoint',
})
vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: { recover: (...args: any[]) => mockRecover(...args) },
}))

import { TrainingHistory } from './TrainingHistory'

const jobs = [
  { id: 'job-1', name: 'distill-a', status: 'completed', loss: 1.5, epochs_completed: 3 },
  { id: 'job-2', name: 'distill-b', status: 'running', loss: 2.25, epochs_completed: 1 },
  { id: 'job-3', name: '', status: 'failed' },
] as any[]

describe('TrainingHistory', () => {
  afterEach(cleanup)

  it('shows empty state when there are no jobs', () => {
    render(<TrainingHistory jobs={[]} />)
    expect(screen.getByText('No training jobs yet')).toBeDefined()
  })

  it('renders job name or id', () => {
    render(<TrainingHistory jobs={jobs} />)
    expect(screen.getByText('distill-a')).toBeDefined()
    expect(screen.getByText('job-3')).toBeDefined()
  })

  it('renders status badges', () => {
    render(<TrainingHistory jobs={jobs} />)
    expect(screen.getByText('completed')).toBeDefined()
    expect(screen.getByText('running')).toBeDefined()
    expect(screen.getByText('failed')).toBeDefined()
  })

  it('renders interrupted status badge', () => {
    render(
      <TrainingHistory
        jobs={[{ id: 'job-9', name: 'interrupted-run', status: 'interrupted' }] as any[]}
      />,
    )
    expect(screen.getByText('interrupted')).toBeDefined()
  })

  it('renders loss with three decimals', () => {
    render(<TrainingHistory jobs={jobs} />)
    expect(screen.getByText('1.500')).toBeDefined()
    expect(screen.getByText('2.250')).toBeDefined()
  })

  it('renders completed epochs', () => {
    render(<TrainingHistory jobs={jobs} />)
    expect(screen.getByText('ep3')).toBeDefined()
  })

  it('does not render loss or epochs when absent', () => {
    render(<TrainingHistory jobs={[jobs[2]]} />)
    expect(screen.queryByText(/^\d+\.\d{3}$/)).toBeNull()
    expect(screen.queryByText(/^ep/)).toBeNull()
  })

  it('shows only first 6 jobs and a +N more line', () => {
    const many = Array.from({ length: 8 }, (_, i) => ({
      id: `job-${i}`,
      name: `name-${i}`,
      status: 'queued',
    })) as any[]
    render(<TrainingHistory jobs={many} />)
    expect(screen.getByText('name-5')).toBeDefined()
    expect(screen.queryByText('name-6')).toBeNull()
    expect(screen.getByText('+2 more')).toBeDefined()
  })

  it('navigates to job detail on row click', () => {
    mockPush.mockClear()
    render(<TrainingHistory jobs={jobs} />)
    const row = screen.getByText('distill-a').closest('[role="button"]') as HTMLElement | null
    expect(row).toBeTruthy()
    row!.click()
    expect(mockPush).toHaveBeenCalledWith('/training/job/job-1')
  })

  it('shows Resume only for interrupted/failed jobs', () => {
    render(<TrainingHistory jobs={jobs} />)
    // jobs fixture: completed, running, failed → one Resume
    expect(screen.getAllByRole('button', { name: 'Resume' })).toHaveLength(1)
  })

  it('resumes an interrupted job and fires onRecovered', async () => {
    const onRecovered = vi.fn()
    mockRecover.mockResolvedValueOnce({
      status: 'recovering',
      recovery_job_id: 'recovery_job-9',
      message: 'Resumed from checkpoint',
    })
    render(
      <TrainingHistory
        jobs={[{ id: 'job-9', name: 'interrupted-run', status: 'interrupted' }] as any[]}
        onRecovered={onRecovered}
      />,
    )
    screen.getByRole('button', { name: 'Resume' }).click()
    await vi.waitFor(() => {
      expect(mockRecover).toHaveBeenCalledWith('job-9')
      expect(onRecovered).toHaveBeenCalledWith(
        expect.objectContaining({ recovery_job_id: 'recovery_job-9' }),
      )
    })
    expect(mockAddToast).toHaveBeenCalledWith('Resumed from checkpoint', 'success')
  })

  it('surfaces the backend reason when resume fails', async () => {
    mockRecover.mockRejectedValueOnce(new Error('No dataset recorded for this job.'))
    render(
      <TrainingHistory
        jobs={[{ id: 'job-9', name: 'interrupted-run', status: 'interrupted' }] as any[]}
      />,
    )
    screen.getByRole('button', { name: 'Resume' }).click()
    await vi.waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith(
        'Could not resume job: No dataset recorded for this job.',
        'error',
      )
    })
  })
})
