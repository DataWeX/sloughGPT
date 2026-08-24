// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { TrainingHealthCard } from './TrainingHealthCard'
import type { Checkpoint } from '@/lib/souls-controller'

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return { name: 'test', soul: 'test', ...overrides }
}

describe('TrainingHealthCard', () => {
  afterEach(cleanup)
  it('shows no-data state for empty checkpoints', () => {
    const { container } = render(<TrainingHealthCard checkpoints={[]} />)
    expect(container.textContent).toContain('No training data yet')
  })

  it('shows no-data when only 1 checkpoint with loss', () => {
    render(<TrainingHealthCard checkpoints={[mkCp({ loss: 1.0 })]} />)
    expect(screen.getAllByText('No data').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Need at least 2/).length).toBeGreaterThanOrEqual(1)
  })

  it('detects improving trend', () => {
    render(
      <TrainingHealthCard
        checkpoints={[
          mkCp({ name: 'a', loss: 3.0 }),
          mkCp({ name: 'b', loss: 2.0 }),
          mkCp({ name: 'c', loss: 1.0 }),
        ]}
      />
    )
    expect(screen.getAllByText('Improving').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Loss trending down/).length).toBeGreaterThanOrEqual(1)
  })

  it('detects diverging trend', () => {
    render(
      <TrainingHealthCard
        checkpoints={[
          mkCp({ name: 'a', loss: 1.0 }),
          mkCp({ name: 'b', loss: 2.0 }),
          mkCp({ name: 'c', loss: 3.0 }),
        ]}
      />
    )
    expect(screen.getAllByText('Diverging').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Loss trending up/).length).toBeGreaterThanOrEqual(1)
  })

  it('detects stagnant trend', () => {
    render(
      <TrainingHealthCard
        checkpoints={[
          mkCp({ name: 'a', loss: 2.0 }),
          mkCp({ name: 'b', loss: 2.01 }),
          mkCp({ name: 'c', loss: 2.005 }),
        ]}
      />
    )
    expect(screen.getAllByText('Stagnant').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Loss flat/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows best loss', () => {
    render(
      <TrainingHealthCard
        checkpoints={[mkCp({ loss: 2.0 }), mkCp({ loss: 1.0 })]}
      />
    )
    expect(screen.getAllByText(/Best loss: 1\.0000/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows best loss from many checkpoints', () => {
    render(
      <TrainingHealthCard
        checkpoints={[
          mkCp({ name: 'a', loss: 5.0 }),
          mkCp({ name: 'b', loss: 3.0 }),
          mkCp({ name: 'c', loss: 1.0 }),
        ]}
      />
    )
    expect(screen.getAllByText(/Best loss: 1\.0000/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows avg quality when checkpoints have quality data', () => {
    render(
      <TrainingHealthCard
        checkpoints={[
          mkCp({ name: 'a', loss: 3.0, avg_quality: 4.2 }),
          mkCp({ name: 'b', loss: 2.0, avg_quality: 3.8 }),
          mkCp({ name: 'c', loss: 1.0, avg_quality: 4.5 }),
        ]}
      />
    )
    expect(screen.getAllByText(/Data quality: 4\.2\/5/).length).toBeGreaterThanOrEqual(1)
  })

  it('omits quality display when no quality data', () => {
    render(
      <TrainingHealthCard
        checkpoints={[mkCp({ loss: 2.0 }), mkCp({ loss: 1.0 })]}
      />
    )
    expect(screen.queryByText(/Data quality/)).toBeNull()
  })

  it('handles two checkpoints with same loss as stagnant', () => {
    render(
      <TrainingHealthCard
        checkpoints={[mkCp({ loss: 2.0 }), mkCp({ loss: 2.0 })]}
      />
    )
    expect(screen.getAllByText('Stagnant').length).toBeGreaterThanOrEqual(1)
  })

  it('shows no-data for checkpoints without loss', () => {
    render(
      <TrainingHealthCard checkpoints={[mkCp({}), mkCp({ name: 'b' })]} />
    )
    expect(screen.getAllByText('No data').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Need at least 2/).length).toBeGreaterThanOrEqual(1)
  })
})
