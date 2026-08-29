/**
 * Tests for ShellPanel component.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@/hooks/useShell', () => ({
  useShell: vi.fn(),
}))

import { useShell } from '@/hooks/useShell'
import { ShellPanel } from './ShellPanel'

const mockUseShell = vi.mocked(useShell)

function createMockShell(overrides: Record<string, unknown> = {}) {
  return {
    state: { lines: [], isRunning: false, exitCode: null, error: null },
    execute: vi.fn().mockResolvedValue(undefined),
    clear: vi.fn(),
    cancel: vi.fn(),
    ...overrides,
  }
}

describe('ShellPanel', () => {
  afterEach(cleanup)
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders placeholder when no lines and not running', () => {
    mockUseShell.mockReturnValue(createMockShell())
    render(<ShellPanel />)
    expect(screen.getByText('Type a command...')).toBeDefined()
  })

  it('renders custom placeholder', () => {
    mockUseShell.mockReturnValue(createMockShell())
    render(<ShellPanel placeholder="Custom prompt" />)
    expect(screen.getByText('Custom prompt')).toBeDefined()
  })

  it('has correct ARIA attributes on output', () => {
    mockUseShell.mockReturnValue(createMockShell())
    render(<ShellPanel />)
    const output = screen.getByTestId('shell-output')
    expect(output).toHaveAttribute('role', 'log')
    expect(output).toHaveAttribute('aria-label', 'Shell output')
  })

  it('has aria-label on input', () => {
    mockUseShell.mockReturnValue(createMockShell())
    render(<ShellPanel />)
    const input = screen.getByTestId('shell-input')
    expect(input).toHaveAttribute('aria-label', 'Shell command input')
  })

  it('renders output lines', () => {
    mockUseShell.mockReturnValue(createMockShell({
      state: {
        lines: [
          { index: 0, text: 'hello' },
          { index: 1, text: 'world' },
        ],
        isRunning: false,
        exitCode: 0,
        error: null,
      },
    }))
    render(<ShellPanel />)
    expect(screen.getByText('hello')).toBeDefined()
    expect(screen.getByText('world')).toBeDefined()
  })

  it('renders running indicator when isRunning', () => {
    mockUseShell.mockReturnValue(createMockShell({
      state: { lines: [], isRunning: true, exitCode: null, error: null },
    }))
    render(<ShellPanel />)
    expect(screen.getByTestId('shell-running')).toBeDefined()
  })

  it('renders error message', () => {
    mockUseShell.mockReturnValue(createMockShell({
      state: { lines: [], isRunning: false, exitCode: 1, error: 'Command failed' },
    }))
    render(<ShellPanel />)
    expect(screen.getByTestId('shell-error')).toHaveTextContent('Command failed')
  })

  it('renders exit code badge on success', () => {
    mockUseShell.mockReturnValue(createMockShell({
      state: { lines: [], isRunning: false, exitCode: 0, error: null },
    }))
    render(<ShellPanel />)
    expect(screen.getByTestId('shell-exit-code')).toHaveTextContent('exit 0')
  })

  it('renders exit code badge on failure', () => {
    mockUseShell.mockReturnValue(createMockShell({
      state: { lines: [], isRunning: false, exitCode: 1, error: null },
    }))
    render(<ShellPanel />)
    expect(screen.getByTestId('shell-exit-code')).toHaveTextContent('exit 1')
  })

  it('submits command on Enter', async () => {
    const execute = vi.fn().mockResolvedValue(undefined)
    mockUseShell.mockReturnValue(createMockShell({ execute }))
    render(<ShellPanel />)

    const input = screen.getByTestId('shell-input')
    fireEvent.change(input, { target: { value: 'echo test' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(execute).toHaveBeenCalledWith('echo test')
  })

  it('does not submit empty commands', () => {
    const execute = vi.fn()
    mockUseShell.mockReturnValue(createMockShell({ execute }))
    render(<ShellPanel />)

    const input = screen.getByTestId('shell-input')
    fireEvent.change(input, { target: { value: '   ' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(execute).not.toHaveBeenCalled()
  })

  it('does not submit while running', () => {
    const execute = vi.fn()
    mockUseShell.mockReturnValue(createMockShell({
      execute,
      state: { lines: [], isRunning: true, exitCode: null, error: null },
    }))
    render(<ShellPanel />)

    const input = screen.getByTestId('shell-input')
    fireEvent.change(input, { target: { value: 'echo test' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(execute).not.toHaveBeenCalled()
  })

  it('clears input after submission', () => {
    const execute = vi.fn().mockResolvedValue(undefined)
    mockUseShell.mockReturnValue(createMockShell({ execute }))
    render(<ShellPanel />)

    const input = screen.getByTestId('shell-input') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'echo test' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(input.value).toBe('')
  })

  it('disables input while running', () => {
    mockUseShell.mockReturnValue(createMockShell({
      state: { lines: [], isRunning: true, exitCode: null, error: null },
    }))
    render(<ShellPanel />)

    const input = screen.getByTestId('shell-input')
    expect(input).toHaveProperty('disabled', true)
  })

  it('calls clear on Ctrl+L', () => {
    const clear = vi.fn()
    mockUseShell.mockReturnValue(createMockShell({ clear }))
    render(<ShellPanel />)

    const input = screen.getByTestId('shell-input')
    fireEvent.keyDown(input, { key: 'l', ctrlKey: true })

    expect(clear).toHaveBeenCalled()
  })

  it('navigates history with ArrowUp', () => {
    const execute = vi.fn().mockResolvedValue(undefined)
    mockUseShell.mockReturnValue(createMockShell({ execute }))
    render(<ShellPanel />)

    const input = screen.getByTestId('shell-input') as HTMLInputElement

    // Execute two commands to build history
    fireEvent.change(input, { target: { value: 'first' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    fireEvent.change(input, { target: { value: 'second' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    // ArrowUp should recall 'second'
    fireEvent.keyDown(input, { key: 'ArrowUp' })
    expect(input.value).toBe('second')

    // ArrowUp again should recall 'first'
    fireEvent.keyDown(input, { key: 'ArrowUp' })
    expect(input.value).toBe('first')
  })

  it('clears history with ArrowDown from history', () => {
    const execute = vi.fn().mockResolvedValue(undefined)
    mockUseShell.mockReturnValue(createMockShell({ execute }))
    render(<ShellPanel />)

    const input = screen.getByTestId('shell-input') as HTMLInputElement

    fireEvent.change(input, { target: { value: 'cmd' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    // Go into history
    fireEvent.keyDown(input, { key: 'ArrowUp' })
    expect(input.value).toBe('cmd')

    // ArrowDown back to end clears input
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    expect(input.value).toBe('')
  })

  it('caps visible lines to maxVisibleLines', () => {
    const lines = Array.from({ length: 20 }, (_, i) => ({ index: i, text: `line${i}` }))
    mockUseShell.mockReturnValue(createMockShell({
      state: { lines, isRunning: false, exitCode: 0, error: null },
    }))
    render(<ShellPanel maxVisibleLines={5} />)

    // Should only show last 5 lines
    expect(screen.getByText('line15')).toBeDefined()
    expect(screen.getByText('line19')).toBeDefined()
    expect(screen.queryByText('line0')).toBeNull()
    expect(screen.queryByText('line14')).toBeNull()
  })
})
