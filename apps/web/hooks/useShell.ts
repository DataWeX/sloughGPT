'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { shellExec, shellExecStream, type ShellExecResult } from '@/lib/shell-controller'

export interface ShellLine {
  text: string
  index: number
  timestamp: number
}

export interface ShellState {
  lines: ShellLine[]
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
    isRunning: false,
    exitCode: null,
    error: null,
  })
  const abortRef = useRef<AbortController | null>(null)

  const clear = useCallback(() => {
    abortRef.current?.abort()
    setState({ lines: [], isRunning: false, exitCode: null, error: null })
  }, [])

  const execute = useCallback(async (command: string, stream = true) => {
    if (!command.trim()) return

    abortRef.current?.abort()
    abortRef.current = new AbortController()

    setState({ lines: [], isRunning: true, exitCode: null, error: null })

    if (stream) {
      const signal = abortRef.current.signal
      await shellExecStream(
        command,
        {
          onLine: (line, index) => {
            if (signal.aborted) return
            setState(prev => ({
              ...prev,
              lines: capLines([...prev.lines, { text: line, index, timestamp: Date.now() }], maxLines),
            }))
          },
          onComplete: (exitCode) => {
            if (signal.aborted) return
            setState(prev => ({
              ...prev,
              isRunning: false,
              exitCode,
              error: null,
            }))
          },
          onError: (error) => {
            if (signal.aborted) return
            setState(prev => ({
              ...prev,
              isRunning: false,
              error,
            }))
          },
        },
        signal,
      )
    } else {
      try {
        const result: ShellExecResult = await shellExec(command, 30000, abortRef.current.signal)
        const lines = capLines(result.output.split('\n').filter((_, i, arr) =>
          i < arr.length - 1 || arr[arr.length - 1] !== ''
        ).map((text, i) => ({
          text,
          index: i,
          timestamp: Date.now(),
        })), maxLines)
        setState({
          lines,
          isRunning: false,
          exitCode: result.exit_code ?? 1,
          error: null,
        })
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          setState(prev => ({ ...prev, isRunning: false }))
          return
        }
        setState(prev => ({
          ...prev,
          isRunning: false,
          error: err instanceof Error ? err.message : 'Unknown error',
        }))
      }
    }
  }, [maxLines])

  const cancel = useCallback(() => {
    abortRef.current?.abort()
    setState(prev => ({ ...prev, isRunning: false }))
  }, [])

  useEffect(() => {
    return () => { abortRef.current?.abort() }
  }, [])

  return { state, execute, clear, cancel }
}
