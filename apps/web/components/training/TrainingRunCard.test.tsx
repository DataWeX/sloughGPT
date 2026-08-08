// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TrainingRunCard } from './TrainingRunCard'
import type { Checkpoint } from '@/lib/souls-controller'

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return {
    name: 'test-checkpoint',
    soul: 'test-soul',
    loss: 2.5,
    model_type: 'slonet',
    ...overrides,
  }
}

describe('TrainingRunCard', () => {
  it('renders checkpoint name and loss', () => {
    render(<TrainingRunCard checkpoint={mkCp()} index={0} />)
    expect(screen.getAllByText('test-checkpoint').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('2.500').length).toBeGreaterThanOrEqual(1)
  })

  it('shows index number', () => {
    render(<TrainingRunCard checkpoint={mkCp()} index={3} />)
    expect(screen.getAllByText('4').length).toBeGreaterThanOrEqual(1)
  })

  it('shows best badge when isBest', () => {
    render(<TrainingRunCard checkpoint={mkCp()} index={0} isBest />)
    expect(screen.getAllByText('best').length).toBeGreaterThanOrEqual(1)
  })

  it('does not render expanded content by default', () => {
    render(<TrainingRunCard checkpoint={mkCp()} index={0} />)
    expect(screen.queryByText('Soul')).toBeNull()
  })

  it('renders data-testid attribute', () => {
    render(<TrainingRunCard checkpoint={mkCp()} index={0} />)
    expect(screen.getAllByTestId('training-run-card').length).toBeGreaterThanOrEqual(1)
  })
})
