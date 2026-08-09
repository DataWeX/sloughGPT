// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { BestCheckpointCard } from './BestCheckpointCard'
import type { Checkpoint } from '@/lib/souls-controller'

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return { name: 'test', soul: 'test', ...overrides }
}

describe('BestCheckpointCard', () => {
  it('returns null when no checkpoints with loss', () => {
    const { container } = render(<BestCheckpointCard checkpoints={[mkCp()]} />)
    expect(container.innerHTML).toBe('')
  })

  it('returns null for empty array', () => {
    const { container } = render(<BestCheckpointCard checkpoints={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('finds checkpoint with lowest loss', () => {
    render(
      <BestCheckpointCard
        checkpoints={[
          mkCp({ name: 'alpha', loss: 2.5 }),
          mkCp({ name: 'beta', loss: 1.0 }),
          mkCp({ name: 'gamma', loss: 3.0 }),
        ]}
      />
    )
    expect(screen.getByText('Best checkpoint')).toBeDefined()
    expect(screen.getByText(/beta/)).toBeDefined()
  })

  it('prefers val_loss over train_loss', () => {
    render(
      <BestCheckpointCard
        checkpoints={[
          mkCp({ name: 'a', loss: 0.5, final_val_loss: 3.0 }),
          mkCp({ name: 'b', loss: 2.0, final_val_loss: 1.0 }),
        ]}
      />
    )
    expect(screen.getByText(/val_loss 1\.0000/)).toBeDefined()
  })

  it('boosts Good verdict', () => {
    render(
      <BestCheckpointCard
        checkpoints={[
          mkCp({ name: 'plain', loss: 2.0 }),
          mkCp({ name: 'good', loss: 1.5, verdict: 'Good' }),
        ]}
      />
    )
    expect(screen.getAllByText('Best checkpoint').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Good').length).toBeGreaterThanOrEqual(1)
  })

  it('boosts Excellent verdict even more', () => {
    render(
      <BestCheckpointCard
        checkpoints={[
          mkCp({ name: 'plain', loss: 1.0 }),
          mkCp({ name: 'excellent', loss: 1.2, verdict: 'Excellent' }),
        ]}
      />
    )
    expect(screen.getAllByText('Best checkpoint').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Excellent').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onLoad with checkpoint name on click', () => {
    const onLoad = vi.fn()
    render(
      <BestCheckpointCard
        checkpoints={[mkCp({ name: 'best', loss: 1.0 })]}
        onLoad={onLoad}
      />
    )
    fireEvent.click(screen.getByText('Use this model'))
    expect(onLoad).toHaveBeenCalledWith('best')
  })

  it('shows duration and epochs when available', () => {
    render(
      <BestCheckpointCard
        checkpoints={[mkCp({ name: 'a', loss: 1.0, epochs_trained: 5, training_duration_s: 120 })]}
      />
    )
    expect(screen.getByText('5 epochs')).toBeDefined()
    expect(screen.getByText('2m 0s')).toBeDefined()
  })

  it('shows loss diff when multiple checkpoints', () => {
    render(
      <BestCheckpointCard
        checkpoints={[
          mkCp({ name: 'best', loss: 1.0 }),
          mkCp({ name: 'second', loss: 2.0 }),
        ]}
      />
    )
    expect(screen.getAllByText(/lower loss than next best/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows Loaded when already loaded', () => {
    render(
      <BestCheckpointCard
        checkpoints={[mkCp({ name: 'a', loss: 1.0, is_loaded: true })]}
        onLoad={vi.fn()}
      />
    )
    expect(screen.getByText('Loaded')).toBeDefined()
  })

  it('ignores zero or negative loss', () => {
    render(
      <BestCheckpointCard
        checkpoints={[
          mkCp({ name: 'zero', loss: 0 }),
          mkCp({ name: 'neg', loss: -1 }),
          mkCp({ name: 'valid', loss: 2.0 }),
        ]}
      />
    )
    expect(screen.getByText(/Loss 2\.0000|loss 2\.0000/)).toBeDefined()
  })
})
