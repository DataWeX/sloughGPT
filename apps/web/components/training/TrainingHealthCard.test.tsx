// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TrainingHealthCard } from './TrainingHealthCard'
import type { Checkpoint } from '@/lib/souls-controller'

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return { name: 'test', soul: 'test', ...overrides }
}

describe('TrainingHealthCard', () => {
  it('returns null for empty checkpoints', () => {
    const { container } = render(<TrainingHealthCard checkpoints={[]} />)
    expect(container.innerHTML).toBe('')
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
})
