// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { TrainingCompareCard } from './TrainingCompareCard'
import type { Checkpoint } from '@/lib/souls-controller'

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return {
    name: 'test-cp',
    soul: 'test-soul',
    loss: 2.5,
    ...overrides,
  }
}

describe('TrainingCompareCard', () => {
  it('renders nothing when fewer than 2 checkpoints', () => {
    const { container } = render(<TrainingCompareCard checkpoints={[mkCp()]} />)
    expect(container.querySelector('[data-testid="training-compare-card"]')).toBeNull()
  })

  it('renders nothing for empty checkpoints', () => {
    const { container } = render(<TrainingCompareCard checkpoints={[]} />)
    expect(container.querySelector('[data-testid="training-compare-card"]')).toBeNull()
  })

  it('renders compare card with 2 checkpoints', () => {
    render(<TrainingCompareCard checkpoints={[mkCp({ name: 'a' }), mkCp({ name: 'b' })]} />)
    expect(screen.getAllByTestId('training-compare-card').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Compare Checkpoints').length).toBeGreaterThanOrEqual(1)
  })

  it('shows both checkpoint names', () => {
    render(<TrainingCompareCard checkpoints={[mkCp({ name: 'alpha' }), mkCp({ name: 'beta' })]} />)
    expect(screen.getAllByText('alpha').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('beta').length).toBeGreaterThanOrEqual(1)
  })

  it('shows loss values', () => {
    render(<TrainingCompareCard checkpoints={[mkCp({ name: 'a', loss: 1.5 }), mkCp({ name: 'b', loss: 2.5 })]} />)
    expect(screen.getAllByText('1.5000').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('2.5000').length).toBeGreaterThanOrEqual(1)
  })

  it('shows loss diff', () => {
    render(<TrainingCompareCard checkpoints={[mkCp({ name: 'a', loss: 1.5 }), mkCp({ name: 'b', loss: 2.5 })]} />)
    expect(screen.getAllByText('-1.0000').length).toBeGreaterThanOrEqual(1)
  })

  it('shows load buttons when onLoad provided', () => {
    render(<TrainingCompareCard checkpoints={[mkCp({ name: 'a' }), mkCp({ name: 'b' })]} onLoad={vi.fn()} />)
    expect(screen.getAllByText('Load A').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Load B').length).toBeGreaterThanOrEqual(1)
  })

  it('shows traits toggle when checkpoints have traits', () => {
    render(<TrainingCompareCard checkpoints={[
      mkCp({ name: 'a', traits: { warmth: 0.8 } }),
      mkCp({ name: 'b', traits: { warmth: 0.6 } }),
    ]} />)
    expect(screen.getAllByText(/Show traits/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows select dropdowns', () => {
    render(<TrainingCompareCard checkpoints={[mkCp({ name: 'a' }), mkCp({ name: 'b' })]} />)
    expect(screen.getAllByTestId('select-a').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByTestId('select-b').length).toBeGreaterThanOrEqual(1)
  })
})
