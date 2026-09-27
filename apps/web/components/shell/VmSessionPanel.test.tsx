import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { VmSessionPanel } from './VmSessionPanel'

const mockStart = vi.fn().mockResolvedValue(undefined)
const mockSendInput = vi.fn().mockResolvedValue(undefined)
const mockDisconnect = vi.fn().mockResolvedValue(undefined)

let mockPhase = 'live'
let mockOutput = ''
let mockError: string | null = null

vi.mock('@/hooks/useVmConsole', () => ({
  useVmConsole: () => ({
    sessionId: 'sid-1',
    phase: mockPhase,
    output: mockOutput,
    error: mockError,
    start: mockStart,
    sendInput: mockSendInput,
    disconnect: mockDisconnect,
  }),
}))

afterEach(() => cleanup())

describe('VmSessionPanel', () => {
  it('starts a session on mount exactly once', () => {
    const { rerender } = render(<VmSessionPanel />)
    expect(mockStart).toHaveBeenCalledTimes(1)
    rerender(<VmSessionPanel />)
    expect(mockStart).toHaveBeenCalledTimes(1)
  })

  it('renders console output', () => {
    mockOutput = 'sloughvm> help\ncommands: help, echo <text>'
    render(<VmSessionPanel />)
    const out = screen.getByTestId('vm-console-output')
    expect(out.textContent).toContain('sloughvm>')
    expect(out.textContent).toContain('commands:')
    mockOutput = ''
  })

  it('shows phase label and error', () => {
    mockPhase = 'error'
    mockError = 'server down'
    render(<VmSessionPanel />)
    expect(screen.getByTestId('vm-session-status').textContent).toBe('Error')
    expect(screen.getByTestId('vm-session-error').textContent).toBe('server down')
    mockPhase = 'live'
    mockError = null
  })

  it('submits typed command with newline', () => {
    render(<VmSessionPanel />)
    const input = screen.getByTestId('vm-session-input') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'help' } })
    const form = input.closest('form')
    expect(form).not.toBeNull()
    fireEvent.submit(form as HTMLFormElement)
    expect(mockSendInput).toHaveBeenCalledWith('help\n')
    expect(input.value).toBe('')
  })

  it('empty input submits a bare newline', () => {
    render(<VmSessionPanel />)
    const input = screen.getByTestId('vm-session-input')
    const form = input.closest('form') as HTMLFormElement
    fireEvent.submit(form)
    expect(mockSendInput).toHaveBeenCalledWith('\n')
  })

  it('ArrowUp recalls submitted commands from history', () => {
    render(<VmSessionPanel />)
    const input = screen.getByTestId('vm-session-input') as HTMLInputElement

    fireEvent.change(input, { target: { value: 'help' } })
    fireEvent.submit(input.closest('form') as HTMLFormElement)
    fireEvent.change(input, { target: { value: 'ls' } })
    fireEvent.submit(input.closest('form') as HTMLFormElement)

    fireEvent.keyDown(input, { key: 'ArrowUp' })
    expect(input.value).toBe('ls')
    fireEvent.keyDown(input, { key: 'ArrowUp' })
    expect(input.value).toBe('help')
    fireEvent.keyDown(input, { key: 'ArrowUp' })
    expect(input.value).toBe('help')
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    expect(input.value).toBe('ls')
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    expect(input.value).toBe('')
  })

  it('ArrowUp with empty history does nothing', () => {
    render(<VmSessionPanel />)
    const input = screen.getByTestId('vm-session-input') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'draft' } })
    fireEvent.keyDown(input, { key: 'ArrowUp' })
    expect(input.value).toBe('draft')
  })

  it('blank submissions are not recorded in history', () => {
    render(<VmSessionPanel />)
    const input = screen.getByTestId('vm-session-input') as HTMLInputElement
    fireEvent.submit(input.closest('form') as HTMLFormElement)
    fireEvent.keyDown(input, { key: 'ArrowUp' })
    expect(input.value).toBe('')
  })

  it('restart button starts a new session', () => {
    render(<VmSessionPanel />)
    expect(mockStart).toHaveBeenCalledTimes(1)
    fireEvent.click(screen.getByText('Restart'))
    expect(mockStart).toHaveBeenCalledTimes(2)
  })

  it('applies className', () => {
    const { container } = render(<VmSessionPanel className="h-96" />)
    expect(container.querySelector('[class*="h-96"]')).not.toBeNull()
  })
})
