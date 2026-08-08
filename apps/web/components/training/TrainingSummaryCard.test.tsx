// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TrainingSummaryCard } from './TrainingSummaryCard'
import type { Checkpoint } from '@/lib/souls-controller'

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return { name: 'test', soul: 'test', ...overrides }
}

describe('TrainingSummaryCard', () => {
  it('returns null for empty checkpoints', () => {
    const { container } = render(<TrainingSummaryCard checkpoints={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('shows total checkpoints even without loss', () => {
    render(<TrainingSummaryCard checkpoints={[mkCp()]} />)
    expect(screen.getAllByText('Total checkpoints').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1').length).toBeGreaterThanOrEqual(1)
  })

  it('shows best and avg loss', () => {
    render(
      <TrainingSummaryCard
        checkpoints={[mkCp({ loss: 1.0 }), mkCp({ loss: 3.0 })]}
      />
    )
    expect(screen.getAllByText('Best loss').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1.0000').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Avg loss').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('2.0000').length).toBeGreaterThanOrEqual(1)
  })

  it('shows loss spread when 2+ checkpoints with loss', () => {
    render(
      <TrainingSummaryCard
        checkpoints={[mkCp({ loss: 1.0 }), mkCp({ loss: 3.0 })]}
      />
    )
    expect(screen.getAllByText('Loss spread').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('2.0000').length).toBeGreaterThanOrEqual(1)
  })

  it('shows total training time', () => {
    render(
      <TrainingSummaryCard
        checkpoints={[mkCp({ loss: 1.0, training_duration_s: 120 }), mkCp({ loss: 2.0, training_duration_s: 60 })]}
      />
    )
    expect(screen.getAllByText('Total training time').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('3m 0s').length).toBeGreaterThanOrEqual(1)
  })

  it('shows fastest run', () => {
    render(
      <TrainingSummaryCard
        checkpoints={[mkCp({ loss: 1.0, training_duration_s: 120 }), mkCp({ loss: 2.0, training_duration_s: 60 })]}
      />
    )
    expect(screen.getAllByText('Fastest run').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1m 0s').length).toBeGreaterThanOrEqual(1)
  })

  it('shows max vocab size', () => {
    render(
      <TrainingSummaryCard
        checkpoints={[mkCp({ loss: 1.0, vocab_size: 100 }), mkCp({ loss: 2.0, vocab_size: 200 })]}
      />
    )
    expect(screen.getAllByText('Max vocab size').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('200').length).toBeGreaterThanOrEqual(1)
  })

  it('formats hours correctly', () => {
    render(
      <TrainingSummaryCard
        checkpoints={[mkCp({ loss: 1.0, training_duration_s: 3720 })]}
      />
    )
    expect(screen.getAllByText('1h 2m').length).toBeGreaterThanOrEqual(1)
  })
})
