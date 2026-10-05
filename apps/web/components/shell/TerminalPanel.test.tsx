/**
 * Tests for TerminalPanel component.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
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
    state: { lines: [], isRunning: false, exitCode: null, error: null },
    execute: vi.fn().mockResolvedValue(undefined),
    clear: vi.fn(),
    cancel: vi.fn(),
    ...overrides,
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

  it('renders output lines', () => {
    mockUseShell.mockReturnValue(
      createMockShell({
        state: {
          lines: [
            { index: 0, text: 'hello' },
            { index: 1, text: 'world' },
          ],
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

  it('renders running indicator when isRunning', () => {
    mockUseShell.mockReturnValue(
      createMockShell({
        state: { lines: [], isRunning: true, exitCode: null, error: null },
      }),
    )
    render(<TerminalPanel />)
    expect(screen.getByTestId('shell-running')).toBeDefined()
  })

  it('renders error message', () => {
    mockUseShell.mockReturnValue(
      createMockShell({
        state: { lines: [], isRunning: false, exitCode: 1, error: 'Command failed' },
      }),
    )
    render(<TerminalPanel />)
    expect(screen.getByTestId('shell-error')).toHaveTextContent('Command failed')
  })

  it('renders exit code badge on success', () => {
    mockUseShell.mockReturnValue(
      createMockShell({
        state: { lines: [], isRunning: false, exitCode: 0, error: null },
      }),
    )
    render(<TerminalPanel />)
    expect(screen.getByTestId('shell-exit-code')).toHaveTextContent('exit 0')
  })

  it('renders exit code badge on failure', () => {
    mockUseShell.mockReturnValue(
      createMockShell({
        state: { lines: [], isRunning: false, exitCode: 1, error: null },
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
        state: { lines: [], isRunning: true, exitCode: null, error: null },
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
        state: { lines: [], isRunning: true, exitCode: null, error: null },
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
    const lines = Array.from({ length: 20 }, (_, i) => ({ index: i, text: `line${i}` }))
    mockUseShell.mockReturnValue(
      createMockShell({
        state: { lines, isRunning: false, exitCode: 0, error: null },
      }),
    )
    render(<TerminalPanel maxVisibleLines={5} />)

    // Should only show last 5 lines
    expect(screen.getByText('line15')).toBeDefined()
    expect(screen.getByText('line19')).toBeDefined()
    expect(screen.queryByText('line0')).toBeNull()
    expect(screen.queryByText('line14')).toBeNull()
  })

  describe('ANSI rendering', () => {
    const ESC = '\u001b'

    it('renders coloured output as styled spans', () => {
      mockUseShell.mockReturnValue(
        createMockShell({
          state: {
            lines: [
              { index: 0, text: `${ESC}[33mWRN${ESC}[0m ok` },
              { index: 1, text: 'plain line' },
            ],
            isRunning: false,
            exitCode: 0,
            error: null,
          },
        }),
      )
      render(<TerminalPanel />)

      // The badge picks up the warning token...
      expect(screen.getByText('WRN').className).toContain('text-warning')
      // ...and an untouched line still lands on the default colour.
      expect(screen.getByText('plain line').className).toContain('text-foreground')
    })

    it('never leaks raw escape codes into the DOM', () => {
      mockUseShell.mockReturnValue(
        createMockShell({
          state: {
            lines: [
              { index: 0, text: `${ESC}[36mhelp header${ESC}[0m` },
              { index: 1, text: `${ESC}[2J${ESC}[H${ESC}[?25lcleared` },
            ],
            isRunning: false,
            exitCode: 0,
            error: null,
          },
        }),
      )
      render(<TerminalPanel />)

      const output = screen.getByTestId('shell-output')
      expect(output.textContent).not.toContain(ESC)
      expect(output.textContent).toBe('help headercleared')
    })

    it('still flags errors whose line begins with an escape', () => {
      mockUseShell.mockReturnValue(
        createMockShell({
          state: {
            lines: [{ index: 0, text: `${ESC}[2mError: boom${ESC}[0m` }],
            isRunning: false,
            exitCode: 1,
            error: null,
          },
        }),
      )
      render(<TerminalPanel />)

      // Scoped to the container on purpose: a fully-styled line is a div and a
      // span with identical text, so getByText would match both.
      const line = screen.getByTestId('shell-output').firstElementChild as HTMLElement

      // The leading escape must not defeat the "Error" prefix check.
      expect(line.className).toContain('text-destructive')
      expect(line.textContent).toBe('Error: boom')
      expect(line.textContent).not.toContain(ESC)
    })

    it('renders plain lines without a wrapper span', () => {
      mockUseShell.mockReturnValue(
        createMockShell({
          state: {
            lines: [{ index: 0, text: 'hello' }],
            isRunning: false,
            exitCode: 0,
            error: null,
          },
        }),
      )
      render(<TerminalPanel />)

      // Exactly one match means no extra nested span was introduced.
      expect(screen.getByText('hello').tagName).toBe('DIV')
    })
  })
})
