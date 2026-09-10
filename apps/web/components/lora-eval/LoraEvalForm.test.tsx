import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { LoraEvalForm } from './LoraEvalForm'

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

describe('LoraEvalForm', () => {
  const defaultProps = {
    adapterPath: 'data/adapter.npz',
    soul: 'assistant',
    running: false,
    onAdapterPathChange: vi.fn(),
    onSoulChange: vi.fn(),
    onRun: vi.fn(),
  }

  it('renders the form title', () => {
    render(<LoraEvalForm {...defaultProps} />)
    expect(screen.getByText('Run Evaluation')).toBeDefined()
  })

  it('renders adapter path and soul inputs', () => {
    render(<LoraEvalForm {...defaultProps} />)
    const inputs = screen.getAllByTestId('input')
    expect(inputs.length).toBeGreaterThanOrEqual(2)
  })

  it('renders the Run Eval button', () => {
    render(<LoraEvalForm {...defaultProps} />)
    expect(screen.getByText('Run Eval')).toBeDefined()
  })

  it('calls onRun when Run Eval is clicked', () => {
    const onRun = vi.fn()
    render(<LoraEvalForm {...defaultProps} onRun={onRun} />)
    fireEvent.click(screen.getByText('Run Eval'))
    expect(onRun).toHaveBeenCalledOnce()
  })

  it('shows Running... when running', () => {
    render(<LoraEvalForm {...defaultProps} running={true} />)
    expect(screen.getByText('Running...')).toBeDefined()
  })

  it('disables Run Eval button when running', () => {
    render(<LoraEvalForm {...defaultProps} running={true} />)
    const btn = screen.getByText('Running...').closest('button')
    expect(btn?.disabled).toBe(true)
  })
})
