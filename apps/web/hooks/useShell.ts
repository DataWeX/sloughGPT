'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { shellExec, shellExecStream, type ShellExecResult } from '@/lib/shell-controller'
import { trackEvent } from '@/lib/dev-log'
import { TerminalEngine, type TerminalRow } from '@/lib/terminal/engine'

export interface ShellLine {
  text: string
  index: number
  timestamp: number
}

export interface ShellState {
  /** Raw line events as received (verbatim, control sequences included). */
  lines: ShellLine[]
  /** Engine-interpreted screen — what the pane renders. */
  rows: TerminalRow[]
  isRunning: boolean
  exitCode: number | null
  error: string | null
}

const DEFAULT_MAX_LINES = 1000

function capLines(lines: ShellLine[], max: number): ShellLine[] {
  if (lines.length <= max) return lines
  return lines.slice(-max)
}

/**
 * Hook for executing shell commands with optional streaming output.
 *
 * @param maxLines - Maximum lines to keep in state (default 1000). Older lines are discarded.
 *
 * @example
 * ```tsx
 * function ShellPanel() {
 *   const { execute, state, clear } = useShell()
 *
 *   return (
 *     <div>
 *       <input onKeyDown={(e) => {
 *         if (e.key === 'Enter') execute(e.currentTarget.value)
 *       }} />
 *       {state.lines.map(l => <div key={l.index}>{l.text}</div>)}
 *     </div>
 *   )
 * }
 * ```
 */
export function useShell(maxLines = DEFAULT_MAX_LINES) {
  const [state, setState] = useState<ShellState>({
    lines: [],
    rows: [],
    isRunning: false,
    exitCode: null,
    error: null,
  })
  const abortRef = useRef<AbortController | null>(null)
  const engineRef = useRef<TerminalEngine | null>(null)
  const getEngine = useCallback(() => {
    if (!engineRef.current) engineRef.current = new TerminalEngine({ maxLines })
    return engineRef.current
  }, [maxLines])

  const clear = useCallback(() => {
    abortRef.current?.abort()
    engineRef.current?.reset()
    setState({ lines: [], rows: [], isRunning: false, exitCode: null, error: null })
  }, [])

  const execute = useCallback(
    async (command: string, stream = true) => {
      if (!command.trim()) return

      abortRef.current?.abort()
      abortRef.current = new AbortController()

      const engine = getEngine()
      engine.reset()
      setState({ lines: [], rows: [], isRunning: true, exitCode: null, error: null })
      trackEvent('shell_command_started', { command_length: command.length })

      if (stream) {
        const signal = abortRef.current.signal
        await shellExecStream(
          command,
          {
            onLine: (line, index) => {
              if (signal.aborted) return
              engine.writeLine(line)
              setState((prev) => ({
                ...prev,
                lines: capLines(
                  [...prev.lines, { text: line, index, timestamp: Date.now() }],
                  maxLines,
                ),
                rows: engine.snapshot(),
              }))
            },
            onComplete: (exitCode) => {
              if (signal.aborted) return
              trackEvent('shell_command_completed', { exit_code: exitCode })
              setState((prev) => ({
                ...prev,
                isRunning: false,
                exitCode,
                error: null,
              }))
            },
            onError: (error) => {
              if (signal.aborted) return
              trackEvent('shell_command_error', { error })
              setState((prev) => ({
                ...prev,
                isRunning: false,
                error,
              }))
            },
          },
          signal,
          // Engine tier: the server passes escape sequences through verbatim
          // so rows carry SGR/cursor/clear semantics (sanitized tier stays the
          // default for every other consumer).
          { raw: true },
        )
      } else {
        try {
          const result: ShellExecResult = await shellExec(command, 30000, abortRef.current.signal)
          const texts = result.output
            .split('\n')
            .filter((_, i, arr) => i < arr.length - 1 || arr[arr.length - 1] !== '')
          const lines = texts.map((text, i) => {
            engine.writeLine(text)
            return { text, index: i, timestamp: Date.now() }
          })
          trackEvent('shell_command_completed', { exit_code: result.exit_code ?? 1 })
          setState({
            lines: capLines(lines, maxLines),
            rows: engine.snapshot(),
            isRunning: false,
            exitCode: result.exit_code ?? 1,
            error: null,
          })
        } catch (err) {
          if (err instanceof DOMException && err.name === 'AbortError') {
            setState((prev) => ({ ...prev, isRunning: false }))
            return
          }
          trackEvent('shell_command_error', {
            error: err instanceof Error ? err.message : 'Unknown error',
          })
          setState((prev) => ({
            ...prev,
            isRunning: false,
            error: err instanceof Error ? err.message : 'Unknown error',
          }))
        }
      }
    },
    [maxLines, getEngine],
  )

  const cancel = useCallback(() => {
    abortRef.current?.abort()
    setState((prev) => ({ ...prev, isRunning: false }))
  }, [])

  useEffect(() => {
    return () => {
      abortRef.current?.abort()
    }
  }, [])

  return { state, execute, clear, cancel }
}
