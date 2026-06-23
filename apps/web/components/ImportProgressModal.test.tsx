// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act, renderHook } from '@testing-library/react'
import React from 'react'

import { ImportProgressModal } from './ImportProgressModal'

const completedJob = {
  id: 'j1',
  source: 'github' as const,
  name: 'test-repo',
  status: 'completed' as const,
  progress: 100,
  message: 'Import complete',
  result: { files_imported: 5, total_chars: 1000 } as any,
}

const failedJob = {
  id: 'j2',
  source: 'url' as const,
  name: 'data.json',
  status: 'failed' as const,
  progress: 30,
  message: 'Download error',
  error: 'Connection timeout',
}

const importingJob = {
  id: 'j3',
  source: 'local' as const,
  name: 'my-data',
  status: 'importing' as const,
  progress: 60,
  message: 'Processing files...',
}

describe('ImportProgressModal', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders title', () => {
    render(<ImportProgressModal open={true} onOpenChange={vi.fn()} jobs={[]} onAllComplete={vi.fn()} />)
    expect(screen.getByText('Importing Datasets')).toBeDefined()
  })

  it('shows importing text when jobs pending', () => {
    render(<ImportProgressModal open={true} onOpenChange={vi.fn()} jobs={[importingJob]} onAllComplete={vi.fn()} />)
    expect(screen.getByText(/Importing 1 dataset/)).toBeDefined()
  })

  it('shows import complete text when no pending jobs', () => {
    render(<ImportProgressModal open={true} onOpenChange={vi.fn()} jobs={[completedJob]} onAllComplete={vi.fn()} />)
    expect(screen.getByText('Import complete!')).toBeDefined()
  })

  it('shows overall progress percentage', () => {
    render(<ImportProgressModal open={true} onOpenChange={vi.fn()} jobs={[completedJob, importingJob]} onAllComplete={vi.fn()} />)
    expect(screen.getByText('50%')).toBeDefined()
  })

  it('renders job list items', () => {
    render(<ImportProgressModal open={true} onOpenChange={vi.fn()} jobs={[completedJob, failedJob, importingJob]} onAllComplete={vi.fn()} />)
    expect(screen.getByText('test-repo')).toBeDefined()
    expect(screen.getByText('data.json')).toBeDefined()
    expect(screen.getByText('my-data')).toBeDefined()
  })

  it('shows error for failed job', () => {
    render(<ImportProgressModal open={true} onOpenChange={vi.fn()} jobs={[failedJob]} onAllComplete={vi.fn()} />)
    expect(screen.getByText('Connection timeout')).toBeDefined()
  })

  it('shows status counts', () => {
    render(<ImportProgressModal open={true} onOpenChange={vi.fn()} jobs={[completedJob, failedJob, importingJob]} onAllComplete={vi.fn()} />)
    expect(screen.getByText(/1 completed/)).toBeDefined()
    expect(screen.getByText(/1 failed/)).toBeDefined()
    expect(screen.getByText(/1 remaining/)).toBeDefined()
  })

  it('shows Done button when all jobs complete', () => {
    render(<ImportProgressModal open={true} onOpenChange={vi.fn()} jobs={[completedJob]} onAllComplete={vi.fn()} />)
    expect(screen.getByText('Done')).toBeDefined()
  })

  it('calls callbacks on Done click', () => {
    const onAllComplete = vi.fn()
    const onOpenChange = vi.fn()
    render(<ImportProgressModal open={true} onOpenChange={onOpenChange} jobs={[completedJob]} onAllComplete={onAllComplete} />)
    fireEvent.click(screen.getByText('Done'))
    expect(onAllComplete).toHaveBeenCalled()
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('hides Done button when jobs are pending', () => {
    render(<ImportProgressModal open={true} onOpenChange={vi.fn()} jobs={[importingJob]} onAllComplete={vi.fn()} />)
    expect(screen.queryByText('Done')).toBeNull()
  })

  it('does not render when closed', () => {
    const { container } = render(<ImportProgressModal open={false} onOpenChange={vi.fn()} jobs={[]} onAllComplete={vi.fn()} />)
    expect(container.innerHTML).toBe('')
  })
})
