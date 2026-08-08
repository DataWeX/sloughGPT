// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TrainingProgress } from './TrainingProgress'

function mkJob(overrides: Record<string, unknown> = {}) {
  return {
    id: '1',
    name: 'Test Job',
    status: 'running',
    progress: 45,
    epoch: 2,
    totalEpochs: 5,
    currentStep: 450,
    totalSteps: 1000,
    loss: 1.23,
    learningRate: 0.001,
    elapsedMs: 60000,
    ...overrides,
  }
}

describe('TrainingProgress', () => {
  it('renders nothing when job is null', () => {
    const { container } = render(<TrainingProgress job={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when job is completed', () => {
    const { container } = render(<TrainingProgress job={mkJob({ status: 'completed' })} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when job is failed', () => {
    const { container } = render(<TrainingProgress job={mkJob({ status: 'failed' })} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders progress card when running', () => {
    render(<TrainingProgress job={mkJob()} />)
    expect(screen.getAllByTestId('training-progress').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Training Progress').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('running').length).toBeGreaterThanOrEqual(1)
  })

  it('shows epoch info', () => {
    render(<TrainingProgress job={mkJob()} />)
    expect(screen.getAllByText('Epoch 2/5').length).toBeGreaterThanOrEqual(1)
  })

  it('shows step info when no epoch', () => {
    render(<TrainingProgress job={mkJob({ epoch: undefined, totalEpochs: undefined })} />)
    expect(screen.getAllByText('Step 450/1,000').length).toBeGreaterThanOrEqual(1)
  })

  it('shows loss and elapsed time', () => {
    render(<TrainingProgress job={mkJob()} />)
    expect(screen.getAllByText('1.2300').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1m 0s').length).toBeGreaterThanOrEqual(1)
  })

  it('shows percentage', () => {
    render(<TrainingProgress job={mkJob({ progress: 75 })} />)
    expect(screen.getAllByText('75%').length).toBeGreaterThanOrEqual(1)
  })

  it('shows queued status', () => {
    render(<TrainingProgress job={mkJob({ status: 'queued' })} />)
    expect(screen.getAllByText('queued').length).toBeGreaterThanOrEqual(1)
  })

  it('shows learning rate when provided', () => {
    const { container } = render(<TrainingProgress job={mkJob()} />)
    expect(container.textContent).toContain('LR:')
  })

  it('hides learning rate when null', () => {
    const { container } = render(<TrainingProgress job={mkJob({ learningRate: null })} />)
    const lrSections = container.querySelectorAll('[class*="muted-foreground/50"]')
    const hasLR = Array.from(lrSections).some(el => el.textContent?.includes('LR:'))
    expect(hasLR).toBe(false)
  })

  it('shows job name', () => {
    render(<TrainingProgress job={mkJob({ name: 'My Training Run' })} />)
    expect(screen.getAllByText('My Training Run').length).toBeGreaterThanOrEqual(1)
  })
})
