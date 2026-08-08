// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TrainingDashboard } from './TrainingDashboard'
import type { Checkpoint } from '@/lib/souls-controller'

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return {
    name: 'test-cp',
    soul: 'test-soul',
    loss: 2.5,
    ...overrides,
  }
}

describe('TrainingDashboard', () => {
  it('renders nothing when no checkpoints', () => {
    const { container } = render(<TrainingDashboard checkpoints={[]} />)
    expect(container.querySelector('[data-testid="training-dashboard"]')).toBeNull()
  })

  it('renders dashboard card', () => {
    render(<TrainingDashboard checkpoints={[mkCp()]} />)
    expect(screen.getAllByTestId('training-dashboard').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Dashboard').length).toBeGreaterThanOrEqual(1)
  })

  it('shows total checkpoints', () => {
    render(<TrainingDashboard checkpoints={[mkCp(), mkCp({ name: 'b' })]} />)
    expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
  })

  it('shows best loss', () => {
    render(<TrainingDashboard checkpoints={[mkCp({ loss: 1.5 }), mkCp({ name: 'b', loss: 3.0 })]} />)
    expect(screen.getAllByText('1.500').length).toBeGreaterThanOrEqual(1)
  })

  it('shows average loss', () => {
    render(<TrainingDashboard checkpoints={[mkCp({ loss: 1.0 }), mkCp({ name: 'b', loss: 3.0 })]} />)
    expect(screen.getAllByText(/avg/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows active badge when checkpoints loaded', () => {
    render(<TrainingDashboard checkpoints={[mkCp({ is_loaded: true })]} />)
    expect(screen.getAllByText('1 active').length).toBeGreaterThanOrEqual(1)
  })

  it('shows model types count', () => {
    render(<TrainingDashboard checkpoints={[mkCp({ model_type: 'slonet' }), mkCp({ name: 'b', model_type: 'lora' })]} />)
    expect(screen.getAllByText('Model Types').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
  })

  it('shows datasets count', () => {
    render(<TrainingDashboard checkpoints={[mkCp({ training_dataset: 'shakespeare' }), mkCp({ name: 'b', training_dataset: 'code' })]} />)
    expect(screen.getAllByText('Datasets').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
  })
})
