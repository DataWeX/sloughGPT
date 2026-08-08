// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TrainingActivity } from './TrainingActivity'
import type { Checkpoint } from '@/lib/souls-controller'

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return {
    name: 'test-cp',
    soul: 'test-soul',
    loss: 2.5,
    ...overrides,
  }
}

describe('TrainingActivity', () => {
  it('renders nothing when no checkpoints', () => {
    const { container } = render(<TrainingActivity checkpoints={[]} />)
    expect(container.querySelector('[data-testid="training-activity"]')).toBeNull()
  })

  it('renders activity card', () => {
    render(<TrainingActivity checkpoints={[mkCp()]} />)
    expect(screen.getAllByTestId('training-activity').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Recent Activity').length).toBeGreaterThanOrEqual(1)
  })

  it('shows checkpoint saved activity', () => {
    render(<TrainingActivity checkpoints={[mkCp({ loss: 1.5 })]} />)
    expect(screen.getAllByText(/Saved test-cp/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/1\.500/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows loaded activity', () => {
    render(<TrainingActivity checkpoints={[mkCp({ is_loaded: true })]} />)
    expect(screen.getAllByText(/Loaded test-cp/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows overfit warning', () => {
    render(<TrainingActivity checkpoints={[mkCp({ verdict: 'overfit' })]} />)
    expect(screen.getAllByText(/overfitting/).length).toBeGreaterThanOrEqual(1)
  })

  it('respects maxItems', () => {
    const cps = Array.from({ length: 10 }, (_, i) => mkCp({ name: `cp-${i}`, loss: i }))
    render(<TrainingActivity checkpoints={cps} maxItems={3} />)
    const items = screen.getAllByTestId('training-activity')
    expect(items.length).toBeGreaterThanOrEqual(1)
  })

  it('shows count badge', () => {
    render(<TrainingActivity checkpoints={[mkCp(), mkCp({ name: 'b' })]} />)
    expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
  })
})
