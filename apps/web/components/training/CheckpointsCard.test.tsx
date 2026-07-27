import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => vi.fn(),
}))

import { CheckpointsCard } from './CheckpointsCard'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

function makeCheckpoints(overrides: Partial<UseTrainingCheckpointsReturn> = {}): UseTrainingCheckpointsReturn {
  return {
    checkpoints: [], checkpointsLoading: false, loadingCheckpoints: false,
    activeCheckpoint: null,
    handleLoadCheckpoint: vi.fn(), handleDeleteCheckpoint: vi.fn(),
    fetchJobs: vi.fn(),
    ...overrides,
  } as UseTrainingCheckpointsReturn
}

const cp = (name: string, overrides: Record<string, any> = {}) => ({
  name, soul: 'friendly', loss: 0.42, epochs_trained: 5,
  description: '', training_dataset: 'shakespeare',
  ...overrides,
})

describe('CheckpointsCard', () => {
  afterEach(cleanup)

  it('returns null when no checkpoints and not loading', () => {
    const { container } = render(
      <CheckpointsCard checkpoints={makeCheckpoints()} loadingTimedOut={false} onRetry={vi.fn()} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    expect(container.innerHTML).toBe('')
  })

  it('renders card title', () => {
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ checkpoints: [cp('v1')] })} loadingTimedOut={false} onRetry={vi.fn()} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    expect(screen.getByText('Trained models')).toBeDefined()
  })

  it('renders checkpoint name', () => {
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ checkpoints: [cp('v1')] })} loadingTimedOut={false} onRetry={vi.fn()} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    expect(screen.getByText('v1')).toBeDefined()
  })

  it('renders loss and epochs', () => {
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ checkpoints: [cp('v1')] })} loadingTimedOut={false} onRetry={vi.fn()} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    expect(screen.getByText(/loss 0\.4200/)).toBeDefined()
    expect(screen.getByText('5 epochs')).toBeDefined()
  })

  it('renders description when provided', () => {
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ checkpoints: [cp('v1', { description: 'My checkpoint' })] })} loadingTimedOut={false} onRetry={vi.fn()} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    expect(screen.getByText('My checkpoint')).toBeDefined()
  })

  it('shows checkmark for active checkpoint', () => {
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ checkpoints: [cp('v1')], activeCheckpoint: 'v1' })} loadingTimedOut={false} onRetry={vi.fn()} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    expect(screen.getByText('✓')).toBeDefined()
  })

  it('hides Load button for active checkpoint', () => {
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ checkpoints: [cp('v1')], activeCheckpoint: 'v1' })} loadingTimedOut={false} onRetry={vi.fn()} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    expect(screen.queryByText('Load')).toBeNull()
  })

  it('shows Load button for inactive checkpoint', () => {
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ checkpoints: [cp('v1')], activeCheckpoint: 'v2' })} loadingTimedOut={false} onRetry={vi.fn()} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    expect(screen.getByText('Load')).toBeDefined()
  })

  it('shows Continue and Del buttons', () => {
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ checkpoints: [cp('v1')] })} loadingTimedOut={false} onRetry={vi.fn()} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    expect(screen.getByText('Continue')).toBeDefined()
    expect(screen.getByText('Del')).toBeDefined()
  })

  it('calls onTest when "Test model" clicked', () => {
    const onTest = vi.fn()
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ checkpoints: [cp('v1')] })} loadingTimedOut={false} onRetry={vi.fn()} onContinue={vi.fn()} onTest={onTest} />
    )
    fireEvent.click(screen.getByText('Test model'))
    expect(onTest).toHaveBeenCalled()
  })

  it('calls onContinue with checkpoint name', () => {
    const onContinue = vi.fn()
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ checkpoints: [cp('v1')] })} loadingTimedOut={false} onRetry={vi.fn()} onContinue={onContinue} onTest={vi.fn()} />
    )
    fireEvent.click(screen.getByText('Continue'))
    expect(onContinue).toHaveBeenCalledWith('v1')
  })

  it('shows skeletons when loading', () => {
    const { container } = render(
      <CheckpointsCard checkpoints={makeCheckpoints({ loadingCheckpoints: true })} loadingTimedOut={false} onRetry={vi.fn()} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    const skeletons = container.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBeGreaterThanOrEqual(1)
  })

  it('shows timeout message and retry button', () => {
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ loadingCheckpoints: true })} loadingTimedOut={true} onRetry={vi.fn()} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    expect(screen.getByText('Taking longer than expected')).toBeDefined()
    expect(screen.getByText('Retry')).toBeDefined()
  })

  it('calls onRetry when retry clicked', () => {
    const onRetry = vi.fn()
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ loadingCheckpoints: true })} loadingTimedOut={true} onRetry={onRetry} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    fireEvent.click(screen.getByText('Retry'))
    expect(onRetry).toHaveBeenCalled()
  })

  it('shows training_dataset when present and not gpt2-generated', () => {
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ checkpoints: [cp('v1', { training_dataset: 'my-data' })] })} loadingTimedOut={false} onRetry={vi.fn()} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    expect(screen.getByText('my-data')).toBeDefined()
  })

  it('hides gpt2-generated training_dataset', () => {
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ checkpoints: [cp('v1', { training_dataset: 'gpt2-generated' })] })} loadingTimedOut={false} onRetry={vi.fn()} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    expect(screen.queryByText('gpt2-generated')).toBeNull()
  })

  it('renders checkpoints in reverse order', () => {
    render(
      <CheckpointsCard checkpoints={makeCheckpoints({ checkpoints: [cp('v1'), cp('v2')] })} loadingTimedOut={false} onRetry={vi.fn()} onContinue={vi.fn()} onTest={vi.fn()} />
    )
    const names = screen.getAllByText(/v[12]/)
    expect(names[0].textContent).toBe('v2')
  })
})
