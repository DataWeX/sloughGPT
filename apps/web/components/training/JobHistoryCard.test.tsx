import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }))
vi.mock('@/lib/controllers', () => ({
  trainingController: { stop: vi.fn(), delete: vi.fn() },
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => vi.fn(),
}))

import { JobHistoryCard } from './JobHistoryCard'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

function makeCheckpoints(overrides: Partial<UseTrainingCheckpointsReturn> = {}): UseTrainingCheckpointsReturn {
  return {
    checkpoints: [], checkpointsLoading: false,
    loadingJobs: false, fetchJobs: vi.fn(),
    handleLoadCheckpoint: vi.fn(), handleDeleteCheckpoint: vi.fn(),
    ...overrides,
  } as UseTrainingCheckpointsReturn
}

function makeJob(overrides: Record<string, any> = {}) {
  return {
    id: 'job-1', name: 'Training run 1', model: 'gpt2', dataset: 'shakespeare',
    status: 'completed', loss: 0.42, current_epoch: 5, epochs: 5,
    created_at: new Date(Date.now() - 300000).toISOString(),
    finished_at: new Date(Date.now() - 60000).toISOString(),
    checkpoint: 'cp-1', status_message: '',
    ...overrides,
  }
}

describe('JobHistoryCard', () => {
  afterEach(cleanup)

  it('returns null when no jobs and not loading', () => {
    const { container } = render(
      <JobHistoryCard allJobs={[]} checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    expect(container.innerHTML).toBe('')
  })

  it('renders card title', () => {
    render(
      <JobHistoryCard allJobs={[makeJob()]} checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    expect(screen.getByText('Job history')).toBeDefined()
  })

  it('renders job name', () => {
    render(
      <JobHistoryCard allJobs={[makeJob()]} checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    expect(screen.getByText('Training run 1')).toBeDefined()
  })

  it('renders job model badge', () => {
    render(
      <JobHistoryCard allJobs={[makeJob()]} checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    expect(screen.getByText('gpt2')).toBeDefined()
  })

  it('renders dataset badge', () => {
    render(
      <JobHistoryCard allJobs={[makeJob()]} checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    expect(screen.getByText('shakespeare')).toBeDefined()
  })

  it('renders loss value', () => {
    render(
      <JobHistoryCard allJobs={[makeJob()]} checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    expect(screen.getByText(/loss 0\.4200/)).toBeDefined()
  })

  it('renders epoch progress', () => {
    render(
      <JobHistoryCard allJobs={[makeJob()]} checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    expect(screen.getByText('epoch 5/5')).toBeDefined()
  })

  it('shows "Done" for completed jobs', () => {
    render(
      <JobHistoryCard allJobs={[makeJob({ status: 'completed' })]} checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    expect(screen.getByText('Done')).toBeDefined()
  })

  it('shows "Use" button for completed jobs with checkpoint', () => {
    render(
      <JobHistoryCard allJobs={[makeJob({ status: 'completed', checkpoint: 'cp-1' })]} checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    expect(screen.getByText('Use')).toBeDefined()
  })

  it('shows "Failed" and "Retry" for failed jobs', () => {
    render(
      <JobHistoryCard allJobs={[makeJob({ status: 'failed', checkpoint: null })]} checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    expect(screen.getByText('Failed')).toBeDefined()
    expect(screen.getByText('Retry')).toBeDefined()
  })

  it('shows "Stop" for running jobs', () => {
    render(
      <JobHistoryCard allJobs={[makeJob({ status: 'running', checkpoint: null, loss: null, finished_at: null })]} checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    expect(screen.getByText('Stop')).toBeDefined()
  })

  it('shows "Stopped" for stopped jobs', () => {
    render(
      <JobHistoryCard allJobs={[makeJob({ status: 'stopped', checkpoint: null })]} checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    expect(screen.getByText('Stopped')).toBeDefined()
  })

  it('shows skeleton when loading jobs', () => {
    const { container } = render(
      <JobHistoryCard allJobs={[]} checkpoints={makeCheckpoints({ loadingJobs: true })} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    const skeletons = container.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBeGreaterThanOrEqual(1)
  })

  it('shows timeout message and retry button', () => {
    render(
      <JobHistoryCard allJobs={[]} checkpoints={makeCheckpoints({ loadingJobs: true })} loadingTimedOut={true} onRetry={vi.fn()} />
    )
    expect(screen.getByText('Taking longer than expected')).toBeDefined()
    expect(screen.getByText('Retry')).toBeDefined()
  })

  it('calls onRetry when retry clicked', () => {
    const onRetry = vi.fn()
    render(
      <JobHistoryCard allJobs={[]} checkpoints={makeCheckpoints({ loadingJobs: true })} loadingTimedOut={true} onRetry={onRetry} />
    )
    fireEvent.click(screen.getByText('Retry'))
    expect(onRetry).toHaveBeenCalled()
  })

  it('renders status_message when provided', () => {
    render(
      <JobHistoryCard allJobs={[makeJob({ status_message: 'Downloading dataset...' })]} checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    expect(screen.getByText('Downloading dataset...')).toBeDefined()
  })

  it('renders relative time for recent jobs', () => {
    const fiveMinAgo = new Date(Date.now() - 5 * 60000).toISOString()
    render(
      <JobHistoryCard allJobs={[makeJob({ created_at: fiveMinAgo, finished_at: null })]} checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} />
    )
    expect(screen.getByText(/5m ago/)).toBeDefined()
  })
})
