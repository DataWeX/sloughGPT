import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const { mockPurge, mockCancel } = vi.hoisted(() => ({
  mockPurge: vi.fn(),
  mockCancel: vi.fn(),
}))

vi.mock('@/lib/system-controller', () => ({
  systemController: {
    purgeExecutorJobs: mockPurge,
    cancelExecutorJob: mockCancel,
  },
}))

import { ExecutorPool } from './ExecutorPool'

const status = {
  initialized: true,
  active_jobs: 1,
  max_workers: 4,
  total_tracked: 5,
  jobs: [
    { job_id: 'run-1', tree_id: null, status: 'running', submitted_at: 0, started_at: 0, completed_at: null, elapsed_s: 12.5, error: null, cancel_requested: false },
    { job_id: 'run-2', tree_id: null, status: 'queued', submitted_at: 0, started_at: null, completed_at: null, elapsed_s: 0, error: null, cancel_requested: false },
    { job_id: 'run-3', tree_id: null, status: 'queued', submitted_at: 0, started_at: null, completed_at: null, elapsed_s: 0, error: null, cancel_requested: false },
    { job_id: 'run-4', tree_id: null, status: 'running', submitted_at: 0, started_at: 0, completed_at: null, elapsed_s: 3, error: null, cancel_requested: true },
    { job_id: 'run-5', tree_id: null, status: 'completed', submitted_at: 0, started_at: 0, completed_at: 1, elapsed_s: 5, error: null, cancel_requested: false },
  ],
} as any

function renderPool(props: Partial<Parameters<typeof ExecutorPool>[0]> = {}) {
  const base = { status, onRefresh: vi.fn() }
  return render(<ExecutorPool {...base} {...props} />)
}

describe('ExecutorPool', () => {
  beforeEach(() => {
    mockPurge.mockReset()
    mockCancel.mockReset()
    mockPurge.mockResolvedValue({ purged: 1 })
    mockCancel.mockResolvedValue({ ok: true })
  })

  afterEach(cleanup)

  it('renders nothing when pool is not initialized', () => {
    const { container } = renderPool({ status: { ...status, initialized: false } })
    expect(container.innerHTML).toBe('')
  })

  it('renders active, workers, tracked and queue counts', () => {
    renderPool()
    expect(screen.getByText('1')).toBeDefined()
    expect(screen.getByText('4')).toBeDefined()
    expect(screen.getByText('5')).toBeDefined()
    expect(screen.getByText('2')).toBeDefined()
  })

  it('renders job ids', () => {
    renderPool()
    expect(screen.getByText('run-1')).toBeDefined()
    expect(screen.getByText('run-3')).toBeDefined()
  })

  it('renders running duration for started jobs', () => {
    renderPool()
    expect(screen.getByText('12.5s')).toBeDefined()
  })

  it('shows purge button when jobs are tracked', () => {
    renderPool()
    expect(screen.getByText('Purge')).toBeDefined()
  })

  it('hides purge button when nothing tracked', () => {
    renderPool({ status: { ...status, total_tracked: 0 } })
    expect(screen.queryByText('Purge')).toBeNull()
  })

  it('purges and refreshes', async () => {
    const onRefresh = vi.fn()
    renderPool({ onRefresh })
    fireEvent.click(screen.getByText('Purge'))
    await waitFor(() => expect(mockPurge).toHaveBeenCalledWith(3600))
    expect(onRefresh).toHaveBeenCalled()
  })

  it('shows cancel buttons only for active non-cancelling jobs', () => {
    renderPool()
    expect(screen.getAllByText('Cancel').length).toBe(3)
  })

  it('cancels a job and refreshes', async () => {
    const onRefresh = vi.fn()
    renderPool({ onRefresh })
    fireEvent.click(screen.getAllByText('Cancel')[0])
    await waitFor(() => expect(mockCancel).toHaveBeenCalledWith('run-1'))
    expect(onRefresh).toHaveBeenCalled()
  })

  it('shows +N more when jobs exceed 4', () => {
    const many = Array.from({ length: 6 }, (_, i) => ({ job_id: `j-${i}`, status: 'queued', elapsed_s: null, cancel_requested: false }))
    renderPool({ status: { ...status, jobs: many } })
    expect(screen.getByText('+2 more')).toBeDefined()
  })
})
