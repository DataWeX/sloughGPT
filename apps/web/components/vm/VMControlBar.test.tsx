/// <reference types="vitest" />
// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { VMControlBar } from './VMControlBar'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Spinner: () => <span data-testid="spinner" />,
}))

describe('VMControlBar', () => {
  const programs = { hello: 'code1', count: 'code2' }
  const defaultProps = {
    programs,
    selectedSource: 'code1',
    onProgramSelect: vi.fn(),
    maxSteps: 5000,
    onMaxStepsChange: vi.fn(),
    keyboardInput: '',
    onKeyboardInputChange: vi.fn(),
    role: 'user',
    onRoleChange: vi.fn(),
    debug: false,
    onDebugChange: vi.fn(),
    running: false,
    onRun: vi.fn(),
    onStep: vi.fn(),
    onClear: vi.fn(),
    hasResult: false,
    showRef: false,
    onToggleRef: vi.fn(),
  }

  it('renders program selector buttons', () => {
    render(<VMControlBar {...defaultProps} />)
    expect(screen.getByText('hello')).toBeDefined()
    expect(screen.getByText('count')).toBeDefined()
  })

  it('renders Run, Step, Clear, and Ref buttons', () => {
    render(<VMControlBar {...defaultProps} />)
    expect(screen.getByText('Run')).toBeDefined()
    expect(screen.getByText('Step')).toBeDefined()
    expect(screen.getByText('Clear')).toBeDefined()
    expect(screen.getByText('Ref')).toBeDefined()
  })

  it('calls onRun when Run is clicked', () => {
    const onRun = vi.fn()
    render(<VMControlBar {...defaultProps} onRun={onRun} />)
    fireEvent.click(screen.getByText('Run'))
    expect(onRun).toHaveBeenCalledOnce()
  })

  it('calls onStep when Step is clicked', () => {
    const onStep = vi.fn()
    render(<VMControlBar {...defaultProps} onStep={onStep} />)
    fireEvent.click(screen.getByText('Step'))
    expect(onStep).toHaveBeenCalledOnce()
  })

  it('calls onProgramSelect when a program button is clicked', () => {
    const onProgramSelect = vi.fn()
    render(<VMControlBar {...defaultProps} onProgramSelect={onProgramSelect} />)
    fireEvent.click(screen.getByText('count'))
    expect(onProgramSelect).toHaveBeenCalledWith('code2')
  })

  it('shows spinner when running', () => {
    render(<VMControlBar {...defaultProps} running />)
    expect(screen.getByTestId('spinner')).toBeDefined()
    expect(screen.getByText('Running')).toBeDefined()
  })

  it('disables Run and Step when running', () => {
    render(<VMControlBar {...defaultProps} running />)
    const runBtn = screen.getByText('Running').closest('button')!
    const stepBtn = screen.getByText('Step')
    expect(runBtn.disabled).toBe(true)
    expect((stepBtn as HTMLButtonElement).disabled).toBe(true)
  })

  it('disables Clear when no result', () => {
    render(<VMControlBar {...defaultProps} hasResult={false} />)
    const clearBtn = screen.getByText('Clear')
    expect((clearBtn as HTMLButtonElement).disabled).toBe(true)
  })

  it('enables Clear when result exists', () => {
    render(<VMControlBar {...defaultProps} hasResult />)
    const clearBtn = screen.getByText('Clear')
    expect((clearBtn as HTMLButtonElement).disabled).toBe(false)
  })

  it('calls onMaxStepsChange when steps input changes', () => {
    const onMaxStepsChange = vi.fn()
    render(<VMControlBar {...defaultProps} onMaxStepsChange={onMaxStepsChange} />)
    const input = screen.getByLabelText('Steps:')
    fireEvent.change(input, { target: { value: '10000' } })
    expect(onMaxStepsChange).toHaveBeenCalledWith(10000)
  })

  it('calls onRoleChange when role select changes', () => {
    const onRoleChange = vi.fn()
    render(<VMControlBar {...defaultProps} onRoleChange={onRoleChange} />)
    fireEvent.change(screen.getByLabelText('VM role'), { target: { value: 'admin' } })
    expect(onRoleChange).toHaveBeenCalledWith('admin')
  })

  it('calls onDebugChange when debug checkbox is toggled', () => {
    const onDebugChange = vi.fn()
    render(<VMControlBar {...defaultProps} onDebugChange={onDebugChange} />)
    fireEvent.click(screen.getByRole('checkbox'))
    expect(onDebugChange).toHaveBeenCalledWith(true)
  })

  it('calls onKeyboardInputChange when keyboard input changes', () => {
    const onKeyboardInputChange = vi.fn()
    render(<VMControlBar {...defaultProps} onKeyboardInputChange={onKeyboardInputChange} />)
    fireEvent.change(screen.getByLabelText('Keyboard input'), { target: { value: 'abc' } })
    expect(onKeyboardInputChange).toHaveBeenCalledWith('abc')
  })

  it('calls onToggleRef when Ref is clicked', () => {
    const onToggleRef = vi.fn()
    render(<VMControlBar {...defaultProps} onToggleRef={onToggleRef} />)
    fireEvent.click(screen.getByText('Ref'))
    expect(onToggleRef).toHaveBeenCalledOnce()
  })
})
