/// <reference types="vitest" />
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { TokenTreeTrainCard } from './TokenTreeTrainCard'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: ({ onChange, ...props }: React.InputHTMLAttributes<HTMLInputElement>) => (
    <input data-testid="input" onChange={onChange} {...props} />
  ),
  Label: ({ children, ...props }: React.PropsWithChildren<React.LabelHTMLAttributes<HTMLLabelElement>>) => (
    <label data-testid="label" {...props}>{children}</label>
  ),
}))

afterEach(() => cleanup())

describe('TokenTreeTrainCard', () => {
  const defaultProps = {
    vocabSize: 256,
    trainTexts: '',
    loading: false,
    onVocabSizeChange: vi.fn(),
    onTrainTextsChange: vi.fn(),
    onTrain: vi.fn(),
  }

  it('renders the card title', () => {
    render(<TokenTreeTrainCard {...defaultProps} />)
    expect(screen.getByText('Train Tree')).toBeDefined()
  })

  it('renders vocab size input', () => {
    render(<TokenTreeTrainCard {...defaultProps} />)
    expect(screen.getByText('Vocab Size')).toBeDefined()
  })

  it('renders train button', () => {
    render(<TokenTreeTrainCard {...defaultProps} />)
    expect(screen.getByText('Train Token Tree')).toBeDefined()
  })

  it('calls onTrain when button is clicked', () => {
    render(<TokenTreeTrainCard {...defaultProps} />)
    fireEvent.click(screen.getByText('Train Token Tree'))
    expect(defaultProps.onTrain).toHaveBeenCalledOnce()
  })

  it('shows Training... when loading', () => {
    render(<TokenTreeTrainCard {...defaultProps} loading />)
    expect(screen.getByText('Training...')).toBeDefined()
  })

  it('disables button when loading', () => {
    render(<TokenTreeTrainCard {...defaultProps} loading />)
    expect(screen.getByText('Training...')).toBeDisabled()
  })
})
