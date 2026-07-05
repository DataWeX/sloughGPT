import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, disabled, ...rest }: any) => (
    <button onClick={onClick} disabled={disabled} {...rest}>{children}</button>
  ),
}))

vi.mock('@/components/ui', () => ({
  IconBrain: () => <span data-testid="icon-brain">brain</span>,
  IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
}))

vi.mock('./LossCurve', () => ({
  LossCurve: ({ data }: any) => <div data-testid="loss-curve" data-points={data.length} />,
}))

import { LearnerTab } from './LearnerTab'

describe('LearnerTab', () => {
  const onTrainStep = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('shows loading skeleton when learnerInfo is null', () => {
    const { container } = render(
      <LearnerTab learnerInfo={null} learnerTraining={false} onTrainStep={onTrainStep} />
    )
    expect(container.querySelector('.animate-pulse')).toBeDefined()
  })

  it('shows tokens and steps', () => {
    render(
      <LearnerTab
        learnerInfo={{ total_tokens_ingested: 5000, train_steps_completed: 42 }}
        learnerTraining={false}
        onTrainStep={onTrainStep}
      />
    )
    expect(screen.getByText('5000')).toBeDefined()
    expect(screen.getByText('42')).toBeDefined()
  })

  it('shows current loss when provided', () => {
    render(
      <LearnerTab
        learnerInfo={{ total_tokens_ingested: 0, train_steps_completed: 0, current_loss: 2.3456 }}
        learnerTraining={false}
        onTrainStep={onTrainStep}
      />
    )
    expect(screen.getByText('2.3456')).toBeDefined()
  })

  it('does not show loss when null', () => {
    render(
      <LearnerTab
        learnerInfo={{ total_tokens_ingested: 0, train_steps_completed: 0 }}
        learnerTraining={false}
        onTrainStep={onTrainStep}
      />
    )
    expect(screen.queryByText(/^\d+\.\d{4}$/)).toBeNull()
  })

  it('shows loss curve when history has >= 2 points', () => {
    render(
      <LearnerTab
        learnerInfo={{
          total_tokens_ingested: 0,
          train_steps_completed: 0,
          loss_history: [
            { step: 1, loss: 2.5, tokens: 100, timestamp: 0 },
            { step: 2, loss: 2.0, tokens: 200, timestamp: 1 },
          ],
        }}
        learnerTraining={false}
        onTrainStep={onTrainStep}
      />
    )
    expect(screen.getByTestId('loss-curve')).toBeDefined()
  })

  it('does not show loss curve when history has fewer than 2 points', () => {
    render(
      <LearnerTab
        learnerInfo={{
          total_tokens_ingested: 0,
          train_steps_completed: 0,
          loss_history: [{ step: 1, loss: 2.5, tokens: 100, timestamp: 0 }],
        }}
        learnerTraining={false}
        onTrainStep={onTrainStep}
      />
    )
    expect(screen.queryByTestId('loss-curve')).toBeNull()
  })

  it('renders Train step button', () => {
    render(
      <LearnerTab
        learnerInfo={{ total_tokens_ingested: 0, train_steps_completed: 0 }}
        learnerTraining={false}
        onTrainStep={onTrainStep}
      />
    )
    expect(screen.getByText('Train step')).toBeDefined()
  })

  it('calls onTrainStep when button clicked', () => {
    render(
      <LearnerTab
        learnerInfo={{ total_tokens_ingested: 0, train_steps_completed: 0 }}
        learnerTraining={false}
        onTrainStep={onTrainStep}
      />
    )
    fireEvent.click(screen.getByText('Train step'))
    expect(onTrainStep).toHaveBeenCalled()
  })

  it('shows Training... spinner when training', () => {
    render(
      <LearnerTab
        learnerInfo={{ total_tokens_ingested: 0, train_steps_completed: 0 }}
        learnerTraining={true}
        onTrainStep={onTrainStep}
      />
    )
    expect(screen.getByText('Training...')).toBeDefined()
  })

  it('disables button when training', () => {
    render(
      <LearnerTab
        learnerInfo={{ total_tokens_ingested: 0, train_steps_completed: 0 }}
        learnerTraining={true}
        onTrainStep={onTrainStep}
      />
    )
    const btn = screen.getByRole('button')
    expect(btn.hasAttribute('disabled')).toBe(true)
  })
})
