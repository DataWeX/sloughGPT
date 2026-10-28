/**
 * Tests for TerminalPanel component.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, act } from '@testing-library/react'
import React from 'react'

vi.mock('@/hooks/useShell', () => ({
  useShell: vi.fn(),
}))

import { useShell } from '@/hooks/useShell'
import { TerminalPanel } from './TerminalPanel'
const ShellPanel = TerminalPanel

const mockUseShell = vi.mocked(useShell)

function createMockShell(overrides: Record<string, unknown> = {}) {
  return {
    state: { lines: [], rows: [], isRunning: false, exitCode: null, error: null },
    execute: vi.fn().mockResolvedValue(undefined),
    clear: vi.fn(),
    cancel: vi.fn(),
    ...overrides,
  }
}

function makeRow(key: number, spans: Array<{ text: string; className?: string }>) {
  return {
    key,
    text: spans.map((s) => s.text).join(''),
    spans: spans.map((s) => ({ text: s.text, className: s.className ?? '' })),
  }
}

describe('TerminalPanel', () => {
  afterEach(cleanup)
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders placeholder when no lines and not running', () => {
    mockUseShell.mockReturnValue(createMockShell())
    render(<TerminalPanel />)
    expect(screen.getByText('Type a command...')).toBeDefined()
  })

  it('renders custom placeholder', () => {
    mockUseShell.mockReturnValue(createMockShell())
    render(<TerminalPanel placeholder="Custom prompt" />)
    expect(screen.getByText('Custom prompt')).toBeDefined()
  })

  it('has correct ARIA attributes on output', () => {
    mockUseShell.mockReturnValue(createMockShell())
    render(<TerminalPanel />)
    const output = screen.getByTestId('shell-output')
    expect(output).toHaveAttribute('role', 'log')
    expect(output).toHaveAttribute('aria-label', 'Shell output')
  })

  it('has aria-label on input', () => {
    mockUseShell.mockReturnValue(createMockShell())
    render(<TerminalPanel />)
    const input = screen.getByTestId('shell-input')
    expect(input).toHaveAttribute('aria-label', 'Shell command input')
  })

  it('renders output rows from the terminal engine', () => {
    mockUseShell.mockReturnValue(
      createMockShell({
        state: {
          lines: [],
          rows: [makeRow(1, [{ text: 'hello' }]), makeRow(2, [{ text: 'world' }])],
          isRunning: false,
          exitCode: 0,
          error: null,
        },
      }),
    )
    render(<TerminalPanel />)
    expect(screen.getByText('hello')).toBeDefined()
    expect(screen.getByText('world')).toBeDefined()
  })

  it('renders styled spans with engine classNames (SGR → design tokens)', () => {
    mockUseShell.mockReturnValue(
      createMockShell({
        state: {
          lines: [],
          rows: [
            makeRow(1, [
              { text: 'ok', className: '' },
              { text: 'bad', className: 'text-term-close' },
            ]),
          ],
          isRunning: false,
          exitCode: 1,
          error: null,
        },
      }),
    )
    render(<TerminalPanel />)
    const styled = screen.getByText('bad')
    expect(styled.className).toContain('text-term-close')
  })

  it('colors Error-prefixed rows as failures (carried-over heuristic)', () => {
    mockUseShell.mockReturnValue(
      createMockShell({
        state: {
          lines: [],
          rows: [
            makeRow(1, [{ text: 'Error: command not found' }]),
            makeRow(2, [{ text: 'all good' }]),
          ],
          isRunning: false,
          exitCode: 1,
          error: null,
        },
      }),
    )
    render(<TerminalPanel />)
    const errRow = screen.getByText('Error: command not found').parentElement
    const okRow = screen.getByText('all good').parentElement
    expect(errRow?.className).toContain('text-term-close')
    expect(okRow?.className).not.toContain('text-term-close')
  })

  it('renders running indicator when isRunning', () => {
    mockUseShell.mockReturnValue(
      createMockShell({
        state: { lines: [], rows: [], isRunning: true, exitCode: null, error: null },
      }),
    )
    render(<TerminalPanel />)
    expect(screen.getByTestId('shell-running')).toBeDefined()
  })

  it('auto-cancels after 60s with a reason argument (loud timeout, not silent)', () => {
    vi.useFakeTimers()
    try {
      const shell = createMockShell({
        state: { lines: [], rows: [], isRunning: true, exitCode: null, error: null },
      })
      mockUseShell.mockReturnValue(shell)
      render(<TerminalPanel />)

      act(() => { vi.advanceTimersByTime(60_000) })

      expect(shell.cancel).toHaveBeenCalledTimes(1)
      expect(shell.cancel).toHaveBeenCalledWith(expect.stringContaining('60s'))
    } finally {
      vi.useRealTimers()
    }
  })

  it('Stop button calls cancel with no reason — user cancel stays silent (timeout-only scope)', () => {
    const shell = createMockShell({
      state: { lines: [], rows: [], isRunning: true, exitCode: null, error: null },
    })
    mockUseShell.mockReturnValue(shell)
    render(<TerminalPanel />)

    fireEvent.click(screen.getByTestId('shell-cancel'))

    expect(shell.cancel).toHaveBeenCalledTimes(1)
    expect(shell.cancel).toHaveBeenCalledWith()
  })

  it('renders error message', () => {
    mockUseShell.mockReturnValue(
      createMockShell({
        state: { lines: [], rows: [], isRunning: false, exitCode: 1, error: 'Command failed' },
      }),
    )
    render(<TerminalPanel />)
    expect(screen.getByTestId('shell-error')).toHaveTextContent('Command failed')
  })

  it('renders exit code badge on success', () => {
    mockUseShell.mockReturnValue(
      createMockShell({
        state: { lines: [], rows: [], isRunning: false, exitCode: 0, error: null },
      }),
    )
    render(<TerminalPanel />)
    expect(screen.getByTestId('shell-exit-code')).toHaveTextContent('exit 0')
  })

  it('renders exit code badge on failure', () => {
    mockUseShell.mockReturnValue(
      createMockShell({
        state: { lines: [], rows: [], isRunning: false, exitCode: 1, error: null },
      }),
    )
    render(<TerminalPanel />)
    expect(screen.getByTestId('shell-exit-code')).toHaveTextContent('exit 1')
  })

  it('submits command on Enter', async () => {
    const execute = vi.fn().mockResolvedValue(undefined)
    mockUseShell.mockReturnValue(createMockShell({ execute }))
    render(<TerminalPanel />)

    const input = screen.getByTestId('shell-input')
    fireEvent.change(input, { target: { value: 'echo test' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(execute).toHaveBeenCalledWith('echo test')
  })

  it('does not submit empty commands', () => {
    const execute = vi.fn()
    mockUseShell.mockReturnValue(createMockShell({ execute }))
    render(<TerminalPanel />)

    const input = screen.getByTestId('shell-input')
    fireEvent.change(input, { target: { value: '   ' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(execute).not.toHaveBeenCalled()
  })

  it('does not submit while running', () => {
    const execute = vi.fn()
    mockUseShell.mockReturnValue(
      createMockShell({
        execute,
        state: { lines: [], rows: [], isRunning: true, exitCode: null, error: null },
      }),
    )
    render(<TerminalPanel />)

    const input = screen.getByTestId('shell-input')
    fireEvent.change(input, { target: { value: 'echo test' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(execute).not.toHaveBeenCalled()
  })

  it('clears input after submission', () => {
    const execute = vi.fn().mockResolvedValue(undefined)
    mockUseShell.mockReturnValue(createMockShell({ execute }))
    render(<TerminalPanel />)

    const input = screen.getByTestId('shell-input') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'echo test' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(input.value).toBe('')
  })

  it('disables input while running', () => {
    mockUseShell.mockReturnValue(
      createMockShell({
        state: { lines: [], rows: [], isRunning: true, exitCode: null, error: null },
      }),
    )
    render(<TerminalPanel />)

    const input = screen.getByTestId('shell-input')
    expect(input).toHaveProperty('disabled', true)
  })

  it('calls clear on Ctrl+L', () => {
    const clear = vi.fn()
    mockUseShell.mockReturnValue(createMockShell({ clear }))
    render(<TerminalPanel />)

    const input = screen.getByTestId('shell-input')
    fireEvent.keyDown(input, { key: 'l', ctrlKey: true })

    expect(clear).toHaveBeenCalled()
  })

  it('navigates history with ArrowUp', () => {
    const execute = vi.fn().mockResolvedValue(undefined)
    mockUseShell.mockReturnValue(createMockShell({ execute }))
    render(<TerminalPanel />)

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
    render(<TerminalPanel />)

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
    const rows = Array.from({ length: 20 }, (_, i) => makeRow(i + 1, [{ text: `line${i}` }]))
    mockUseShell.mockReturnValue(
      createMockShell({
        state: { lines: [], rows, isRunning: false, exitCode: 0, error: null },
      }),
    )
    render(<TerminalPanel maxVisibleLines={5} />)

    // Should only show last 5 lines
    expect(screen.getByText('line15')).toBeDefined()
    expect(screen.getByText('line19')).toBeDefined()
    expect(screen.queryByText('line0')).toBeNull()
    expect(screen.queryByText('line14')).toBeNull()
  })
})
