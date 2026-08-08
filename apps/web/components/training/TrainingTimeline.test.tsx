// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TrainingTimeline } from './TrainingTimeline'
import type { Checkpoint } from '@/lib/souls-controller'

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return {
    name: 'test-checkpoint',
    soul: 'test-soul',
    loss: 2.5,
    ...overrides,
  }
}

describe('TrainingTimeline', () => {
  it('renders nothing when no checkpoints', () => {
    const { container } = render(<TrainingTimeline checkpoints={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders timeline card with checkpoints', () => {
    render(<TrainingTimeline checkpoints={[mkCp(), mkCp({ name: 'cp-2', loss: 1.5 })]} />)
    expect(screen.getAllByTestId('training-timeline').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Timeline').length).toBeGreaterThanOrEqual(1)
  })

  it('shows checkpoint names', () => {
    render(<TrainingTimeline checkpoints={[mkCp(), mkCp({ name: 'alpha' })]} />)
    expect(screen.getAllByText('test-checkpoint').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('alpha').length).toBeGreaterThanOrEqual(1)
  })

  it('shows timeline events', () => {
    render(<TrainingTimeline checkpoints={[mkCp()]} />)
    expect(screen.getAllByTestId('timeline-event').length).toBeGreaterThanOrEqual(1)
  })

  it('shows loaded badge when checkpoint is_loaded', () => {
    render(<TrainingTimeline checkpoints={[mkCp({ is_loaded: true })]} />)
    expect(screen.getAllByText('loaded').length).toBeGreaterThanOrEqual(1)
  })

  it('shows model_type detail', () => {
    render(<TrainingTimeline checkpoints={[mkCp({ model_type: 'slonet', training_dataset: 'shakespeare' })]} />)
    expect(screen.getAllByText('slonet · shakespeare').length).toBeGreaterThanOrEqual(1)
  })

  it('respects maxEvents', () => {
    const cps = Array.from({ length: 10 }, (_, i) => mkCp({ name: `cp-${i}` }))
    render(<TrainingTimeline checkpoints={cps} maxEvents={3} />)
    expect(screen.getAllByTestId('timeline-event').length).toBeGreaterThanOrEqual(3)
  })
})
