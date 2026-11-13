/// <reference types="vitest" />
import '@testing-library/jest-dom/vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, describe, it, expect, vi } from 'vitest'
import { AggregateCard } from './AggregateCard'

afterEach(() => cleanup())

const defaultProps = {
  topK: 10,
  minFeedback: 5,
  aggregating: false,
  onTopKChange: vi.fn(),
  onMinFeedbackChange: vi.fn(),
  onAggregate: vi.fn(),
}

describe('AggregateCard', () => {
  it('renders title', () => {
    render(<AggregateCard {...defaultProps} />)
    expect(screen.getByText('Aggregate Adapters')).toBeInTheDocument()
  })

  it('renders top K input', () => {
    render(<AggregateCard {...defaultProps} />)
    expect(screen.getByDisplayValue('10')).toBeInTheDocument()
  })

  it('renders min feedback input', () => {
    render(<AggregateCard {...defaultProps} />)
    expect(screen.getByDisplayValue('5')).toBeInTheDocument()
  })

  it('renders Aggregate button', () => {
    render(<AggregateCard {...defaultProps} />)
    expect(screen.getByText('Aggregate')).toBeInTheDocument()
  })

  it('shows loading state', () => {
    render(<AggregateCard {...defaultProps} aggregating={true} />)
    expect(screen.getByText('Aggregating...')).toBeInTheDocument()
  })

  it('disables button when aggregating', () => {
    render(<AggregateCard {...defaultProps} aggregating={true} />)
    expect(screen.getByText('Aggregating...')).toBeDisabled()
  })
})
