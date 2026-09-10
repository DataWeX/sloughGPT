import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { KbAddEntryForm } from './KbAddEntryForm'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: ({ onChange, ...props }: any) => <input data-testid="input" onChange={onChange} {...props} />,
  Label: ({ children, ...props }: any) => <label data-testid="label" {...props}>{children}</label>,
  Textarea: ({ onChange, ...props }: any) => <textarea data-testid="textarea" onChange={onChange} {...props} />,
  Slider: () => <div data-testid="slider" />,
}))

afterEach(() => cleanup())

describe('KbAddEntryForm', () => {
  const defaultProps = {
    content: '',
    topic: 'general',
    importance: 0.7,
    loading: false,
    suggestResult: null,
    onContentChange: vi.fn(),
    onTopicChange: vi.fn(),
    onImportanceChange: vi.fn(),
    onAdd: vi.fn(),
    onSuggest: vi.fn(),
  }

  it('renders the form title', () => {
    render(<KbAddEntryForm {...defaultProps} />)
    expect(screen.getByText('Add Knowledge Entry')).toBeDefined()
  })

  it('renders content and topic inputs', () => {
    render(<KbAddEntryForm {...defaultProps} />)
    expect(screen.getAllByTestId('textarea').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByTestId('input').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Add Entry and Suggest Topic buttons', () => {
    render(<KbAddEntryForm {...defaultProps} />)
    expect(screen.getByText('Add Entry')).toBeDefined()
    expect(screen.getByText('Suggest Topic')).toBeDefined()
  })

  it('calls onAdd when Add Entry is clicked', () => {
    const onAdd = vi.fn()
    render(<KbAddEntryForm {...defaultProps} content="test" onAdd={onAdd} />)
    fireEvent.click(screen.getByText('Add Entry'))
    expect(onAdd).toHaveBeenCalledOnce()
  })

  it('disables Add Entry when content is empty', () => {
    render(<KbAddEntryForm {...defaultProps} />)
    const addBtn = screen.getByText('Add Entry')
    expect(addBtn.closest('button')?.disabled).toBe(true)
  })

  it('shows suggested topic when provided', () => {
    render(<KbAddEntryForm {...defaultProps} suggestResult="science" />)
    expect(screen.getByText('Suggested: science')).toBeDefined()
  })

  it('shows Adding... when loading', () => {
    render(<KbAddEntryForm {...defaultProps} content="test" loading={true} />)
    expect(screen.getByText('Adding...')).toBeDefined()
  })
})
