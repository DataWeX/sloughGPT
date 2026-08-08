// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { TrainingOnboarding } from './TrainingOnboarding'

describe('TrainingOnboarding', () => {
  it('renders nothing when hasCheckpoints is true', () => {
    const { container } = render(<TrainingOnboarding hasCheckpoints={true} />)
    expect(container.querySelector('[data-testid="training-onboarding"]')).toBeNull()
  })

  it('renders onboarding card when no checkpoints', () => {
    render(<TrainingOnboarding hasCheckpoints={false} />)
    expect(screen.getAllByTestId('training-onboarding').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Welcome to Training').length).toBeGreaterThanOrEqual(1)
  })

  it('shows all 4 steps', () => {
    render(<TrainingOnboarding hasCheckpoints={false} />)
    expect(screen.getAllByText('Add training data').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Configure & start').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Monitor progress').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Use your model').length).toBeGreaterThanOrEqual(1)
  })

  it('shows description text', () => {
    render(<TrainingOnboarding hasCheckpoints={false} />)
    expect(screen.getAllByText(/Train your own AI model/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows start training button when callback provided', () => {
    render(<TrainingOnboarding hasCheckpoints={false} onStartTraining={vi.fn()} />)
    expect(screen.getAllByText('Start training').length).toBeGreaterThanOrEqual(1)
  })

  it('shows import data button when callback provided', () => {
    render(<TrainingOnboarding hasCheckpoints={false} onImportData={vi.fn()} />)
    expect(screen.getAllByText('Import data').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onStartTraining when clicked', () => {
    const onStartTraining = vi.fn()
    render(<TrainingOnboarding hasCheckpoints={false} onStartTraining={onStartTraining} />)
    const btns = screen.getAllByText('Start training')
    fireEvent.click(btns[btns.length - 1])
    expect(onStartTraining).toHaveBeenCalledTimes(1)
  })

  it('calls onImportData when clicked', () => {
    const onImportData = vi.fn()
    render(<TrainingOnboarding hasCheckpoints={false} onImportData={onImportData} />)
    const btns = screen.getAllByText('Import data')
    fireEvent.click(btns[btns.length - 1])
    expect(onImportData).toHaveBeenCalledTimes(1)
  })
})
