import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { LoraAggregateForm } from './LoraAggregateForm'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: ({ onChange, ...props }: any) => <input data-testid="input" onChange={onChange} {...props} />,
}))

afterEach(() => cleanup())

describe('LoraAggregateForm', () => {
  const defaultProps = {
    topK: 10,
    minFeedback: 5,
    aggregating: false,
    onTopKChange: vi.fn(),
    onMinFeedbackChange: vi.fn(),
    onAggregate: vi.fn(),
  }

  it('renders the form title', () => {
    render(<LoraAggregateForm {...defaultProps} />)
    expect(screen.getByText('Aggregate Adapters')).toBeDefined()
  })

  it('renders Top K and Min Feedback inputs', () => {
    render(<LoraAggregateForm {...defaultProps} />)
    const inputs = screen.getAllByTestId('input')
    expect(inputs.length).toBeGreaterThanOrEqual(2)
  })

  it('renders the Aggregate button', () => {
    render(<LoraAggregateForm {...defaultProps} />)
    expect(screen.getByText('Aggregate')).toBeDefined()
  })

  it('calls onAggregate when Aggregate is clicked', () => {
    const onAggregate = vi.fn()
    render(<LoraAggregateForm {...defaultProps} onAggregate={onAggregate} />)
    fireEvent.click(screen.getByText('Aggregate'))
    expect(onAggregate).toHaveBeenCalledOnce()
  })

  it('shows Aggregating... when aggregating', () => {
    render(<LoraAggregateForm {...defaultProps} aggregating={true} />)
    expect(screen.getByText('Aggregating...')).toBeDefined()
  })

  it('disables Aggregate button when aggregating', () => {
    render(<LoraAggregateForm {...defaultProps} aggregating={true} />)
    const btn = screen.getByText('Aggregating...').closest('button')
    expect(btn?.disabled).toBe(true)
  })
})
