// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TrainingTips } from './TrainingTips'
import type { Checkpoint } from '@/lib/souls-controller'

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return {
    name: 'test-cp',
    soul: 'test-soul',
    loss: 2.5,
    ...overrides,
  }
}

describe('TrainingTips', () => {
  it('renders nothing when no tips apply', () => {
    const { container } = render(<TrainingTips checkpoints={[mkCp(), mkCp({ name: 'b' })]} />)
    expect(container.querySelector('[data-testid="training-tips"]')).toBeNull()
  })

  it('shows get-started tip when no checkpoints', () => {
    render(<TrainingTips checkpoints={[]} />)
    expect(screen.getAllByTestId('training-tips').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Get started').length).toBeGreaterThanOrEqual(1)
  })

  it('shows first-checkpoint tip when only one checkpoint', () => {
    render(<TrainingTips checkpoints={[mkCp({ loss: 1.5 })]} />)
    expect(screen.getAllByText('First checkpoint saved').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/1\.500/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows loss-spread tip when large spread', () => {
    render(<TrainingTips checkpoints={[mkCp({ loss: 0.5 }), mkCp({ name: 'b', loss: 3.0 })]} />)
    expect(screen.getAllByText('Large loss spread detected').length).toBeGreaterThanOrEqual(1)
  })

  it('shows overfit warning', () => {
    render(<TrainingTips checkpoints={[mkCp({ verdict: 'overfit' })]} />)
    expect(screen.getAllByText('Overfitting detected').length).toBeGreaterThanOrEqual(1)
  })

  it('shows training-active tip when training', () => {
    render(<TrainingTips checkpoints={[mkCp()]} isTraining />)
    expect(screen.getAllByText('Training in progress').length).toBeGreaterThanOrEqual(1)
  })

  it('shows ready-to-train tip when dataset selected but no checkpoints', () => {
    render(<TrainingTips checkpoints={[]} hasDataset />)
    expect(screen.getAllByText('Ready to train').length).toBeGreaterThanOrEqual(1)
  })

  it('shows multiple-loaded warning', () => {
    render(<TrainingTips checkpoints={[mkCp({ is_loaded: true }), mkCp({ name: 'b', is_loaded: true })]} />)
    expect(screen.getAllByText('Multiple models loaded').length).toBeGreaterThanOrEqual(1)
  })

  it('shows many-epochs warning', () => {
    render(<TrainingTips checkpoints={[mkCp({ epochs_trained: 25, training_dataset: 'shakespeare' })]} />)
    expect(screen.getAllByText('Many epochs detected').length).toBeGreaterThanOrEqual(1)
  })

  it('limits to 3 tips max', () => {
    render(<TrainingTips checkpoints={[
      mkCp({ loss: 0.1, verdict: 'overfit', is_loaded: true, epochs_trained: 30 }),
      mkCp({ name: 'b', loss: 3.0, is_loaded: true }),
    ]} isTraining />)
    const tips = screen.getAllByTestId('training-tips')
    expect(tips.length).toBeGreaterThanOrEqual(1)
  })
})
