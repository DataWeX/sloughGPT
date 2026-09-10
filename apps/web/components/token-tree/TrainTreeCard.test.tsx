import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { TrainTreeCard } from './TrainTreeCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: (props: any) => <input data-testid="vocab-input" {...props} />,
  Label: ({ children, ...props }: any) => <label data-testid="label" {...props}>{children}</label>,
}))

const defaultProps = {
  vocabSize: 256,
  trainTexts: '',
  loading: false,
  onVocabSizeChange: vi.fn(),
  onTrainTextsChange: vi.fn(),
  onTrain: vi.fn(),
}

describe('TrainTreeCard', () => {
  it('renders the card title', () => {
    render(<TrainTreeCard {...defaultProps} />)
    expect(screen.getByText('Train Tree')).toBeDefined()
  })

  it('renders the vocab size input', () => {
    render(<TrainTreeCard {...defaultProps} />)
    expect(screen.getByTestId('vocab-input')).toBeDefined()
  })

  it('displays the current vocab size value', () => {
    render(<TrainTreeCard {...defaultProps} vocabSize={512} />)
    expect((screen.getByTestId('vocab-input') as HTMLInputElement).value).toBe('512')
  })

  it('renders the train button with correct label', () => {
    render(<TrainTreeCard {...defaultProps} />)
    expect(screen.getByText('Train Token Tree')).toBeDefined()
  })

  it('shows training state when loading', () => {
    render(<TrainTreeCard {...defaultProps} loading={true} />)
    expect(screen.getByText('Training...')).toBeDefined()
  })

  it('renders labels', () => {
    render(<TrainTreeCard {...defaultProps} />)
    expect(screen.getByText('Vocab Size')).toBeDefined()
  })

  it('renders the textarea for training texts', () => {
    render(<TrainTreeCard {...defaultProps} />)
    expect(screen.getByPlaceholderText('Leave empty to use default corpus...')).toBeDefined()
  })
})
