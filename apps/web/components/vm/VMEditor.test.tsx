/// <reference types="vitest" />
// @vitest-environment jsdom
import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, describe, it, expect, vi } from 'vitest'
import { VMEditor } from './VMEditor'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div data-testid="card-header">{children}</div>,
  CardTitle: ({ children }: any) => <div data-testid="card-title">{children}</div>,
  CardContent: ({ children }: any) => <div data-testid="card-content">{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
}))

afterEach(() => cleanup())

describe('VMEditor', () => {
  const defaultProps = {
    source: 'MOV EAX, 42\nHLT',
    onSourceChange: vi.fn(),
    onRun: vi.fn(),
    onStep: vi.fn(),
    onClear: vi.fn(),
    running: false,
    hasResult: false,
  }

  it('renders the card title', () => {
    render(<VMEditor {...defaultProps} />)
    expect(screen.getByText('Assembly Source')).toBeDefined()
  })

  it('renders the source code textarea', () => {
    render(<VMEditor {...defaultProps} />)
    const textarea = screen.getByLabelText('Assembly source code')
    expect(textarea).toBeDefined()
    expect((textarea as HTMLTextAreaElement).value).toBe('MOV EAX, 42\nHLT')
  })

  it('renders Run and Step buttons', () => {
    render(<VMEditor {...defaultProps} />)
    expect(screen.getByText('Run')).toBeDefined()
    expect(screen.getByText('Step')).toBeDefined()
  })

  it('disables buttons when running', () => {
    render(<VMEditor {...defaultProps} running={true} />)
    expect((screen.getByText('Running...') as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByText('Step') as HTMLButtonElement).disabled).toBe(true)
  })

  it('shows line numbers for source code', () => {
    render(<VMEditor {...defaultProps} />)
    expect(screen.getByText('1')).toBeDefined()
    expect(screen.getByText('2')).toBeDefined()
  })

  it('disables Clear when no result', () => {
    render(<VMEditor {...defaultProps} hasResult={false} />)
    expect((screen.getByText('Clear') as HTMLButtonElement).disabled).toBe(true)
  })

  it('calls onRun when Run is clicked', () => {
    const onRun = vi.fn()
    render(<VMEditor {...defaultProps} onRun={onRun} />)
    screen.getByText('Run').click()
    expect(onRun).toHaveBeenCalledOnce()
  })
})
