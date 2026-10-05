'use client'

import { useState, useRef, useEffect, Fragment, type KeyboardEvent } from 'react'
import { cn } from '@sloughgpt/strui'
import { useShell, type ShellLine } from '@/hooks/useShell'
import { parseAnsi, type AnsiSegment } from '@/lib/ansi'

export interface TerminalPanelProps {
  className?: string
  placeholder?: string
  maxVisibleLines?: number
}

/**
 * Per-line parse cache.
 *
 * Streaming re-renders on every appended line, and parsing is ~30us per styled
 * line — re-parsing the whole buffer each render is O(n^2) across a stream
 * (measured: 30ms per render at the 1000-line cap, past the 16ms frame budget).
 *
 * `capLines` spreads the existing line objects into each new array, so a given
 * ShellLine keeps its identity across renders and can be parsed exactly once.
 * WeakMap keys are released when capLines drops the line, so this cannot grow
 * without bound.
 */
const lineCache = new WeakMap<ShellLine, { segments: AnsiSegment[]; isError: boolean }>()

interface LineView {
  segments: AnsiSegment[]
  isError: boolean
}

function analyzeLine(line: ShellLine): LineView {
  const hit = lineCache.get(line)
  if (hit) return hit

  // One tokenize pass yields both the styled segments and the visible text, so
  // the error check does not re-walk the string.
  const segments = parseAnsi(line.text)
  const visible = segments.map((s) => s.text).join('')
  // The REPL returns SGR colour codes; match on the visible text so a leading
  // escape never defeats the error check.
  const isError = visible.startsWith('Error') || visible.startsWith('error')

  const value: LineView = { segments, isError }
  lineCache.set(line, value)
  return value
}

/**
 * Terminal-like shell panel with command input and streaming output.
 *
 * @example
 * ```tsx
 * <ShellPanel className="h-96" />
 * ```
 */
export function TerminalPanel({
  className,
  placeholder = 'Type a command...',
  maxVisibleLines = 500,
}: TerminalPanelProps) {
  const { state, execute, clear, cancel } = useShell()
  const [input, setInput] = useState('')
  const [history, setHistory] = useState<string[]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)
  const [hideExit, setHideExit] = useState(false)
  const outputRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Auto-scroll to bottom
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [state.lines])

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  // Auto-cancel after 60s of running
  const startRef = useRef<number | null>(null)
  useEffect(() => {
    if (state.isRunning) {
      startRef.current = Date.now()
      const timer = setTimeout(() => {
        cancel()
        startRef.current = null
      }, 60_000)
      return () => clearTimeout(timer)
    }
    startRef.current = null
  }, [state.isRunning, cancel])

  // Auto-hide exit code badge after 5s for success, keep showing for errors
  useEffect(() => {
    if (state.exitCode === 0 && !state.isRunning) {
      setHideExit(false)
      const timer = setTimeout(() => setHideExit(true), 5000)
      return () => clearTimeout(timer)
    }
    setHideExit(false)
  }, [state.exitCode, state.isRunning])

  const handleSubmit = () => {
    const cmd = input.trim()
    if (!cmd) return

    setHistory((prev) => [...prev, cmd])
    setHistoryIndex(-1)
    setInput('')
    execute(cmd)
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !state.isRunning) {
      handleSubmit()
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      if (history.length > 0) {
        const newIndex = historyIndex < history.length - 1 ? historyIndex + 1 : historyIndex
        setHistoryIndex(newIndex)
        const val = history[history.length - 1 - newIndex] ?? ''
        setInput(val)
        requestAnimationFrame(() => {
          const len = val.length
          inputRef.current?.setSelectionRange(len, len)
        })
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1
        setHistoryIndex(newIndex)
        const val = history[history.length - 1 - newIndex] ?? ''
        setInput(val)
        requestAnimationFrame(() => {
          const len = val.length
          inputRef.current?.setSelectionRange(len, len)
        })
      } else {
        setHistoryIndex(-1)
        setInput('')
        requestAnimationFrame(() => {
          inputRef.current?.setSelectionRange(0, 0)
        })
      }
    } else if (e.key === 'l' && e.ctrlKey) {
      e.preventDefault()
      clear()
    }
  }

  const visibleLines = state.lines.slice(-maxVisibleLines)

  return (
    <div
      className={cn(
        'flex flex-col rounded-xl overflow-hidden',
        'border border-white/[0.08] shadow-2xl shadow-black/40',
        'bg-background',
        className,
      )}
    >
      {/* macOS title bar */}
      <div className="flex items-center h-11 px-4 bg-muted border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-destructive shadow-[inset_0_-1px_1px_rgba(0,0,0,0.15)] ring-1 ring-black/10" />
          <span className="w-3 h-3 rounded-full bg-warning shadow-[inset_0_-1px_1px_rgba(0,0,0,0.15)] ring-1 ring-black/10" />
          <span className="w-3 h-3 rounded-full bg-success shadow-[inset_0_-1px_1px_rgba(0,0,0,0.15)] ring-1 ring-black/10" />
        </div>
        <div className="flex-1 flex items-center justify-center">
          <span className="text-[11px] font-medium text-muted-foreground select-none">dait</span>
        </div>
        <span
          className={cn(
            'w-1.5 h-1.5 rounded-full',
            state.isRunning
              ? 'bg-warning animate-pulse'
              : state.exitCode !== null && state.exitCode !== 0
                ? 'bg-destructive'
                : 'bg-success',
          )}
        />
      </div>

      {/* Output area */}
      <div
        ref={outputRef}
        className="flex-1 overflow-y-auto px-5 py-4 font-mono text-[12px] leading-[1.7] text-foreground"
        data-testid="shell-output"
        role="log"
        aria-live="polite"
        aria-label="Shell output"
      >
        {visibleLines.length === 0 && !state.isRunning && placeholder && (
          <div className="text-muted-foreground italic">{placeholder}</div>
        )}
        {visibleLines.map((line: ShellLine) => {
          const { segments, isError } = analyzeLine(line)
          // Unstyled lines render as bare text (no wrapper element) so they
          // stay byte-identical to what the tests — and screen readers — expect.
          const bare = segments.length === 1 && segments[0].className === ''

          return (
            <div
              key={line.index}
              className={cn(
                'whitespace-pre-wrap break-all',
                isError ? 'text-destructive' : 'text-foreground',
              )}
            >
              {bare
                ? segments[0].text
                : segments.map((segment, i) =>
                    segment.className ? (
                      <span key={i} className={segment.className}>
                        {segment.text}
                      </span>
                    ) : (
                      // Fragment, not span: an unstyled run must not add a DOM
                      // node, or getByText would match both it and the parent.
                      <Fragment key={i}>{segment.text}</Fragment>
                    ),
                  )}
            </div>
          )
        })}
        {state.isRunning && (
          <div className="flex items-center gap-2 text-warning" data-testid="shell-running">
            <span className="w-1.5 h-1.5 rounded-full bg-warning animate-pulse" />
            <span className="text-[10px]">executing...</span>
          </div>
        )}
        {state.error && (
          <div
            className="mt-2 flex items-start gap-2 text-destructive bg-destructive/[0.08] rounded-md px-3 py-2"
            data-testid="shell-error"
          >
            <span className="shrink-0 text-[10px] font-bold">!</span>
            <span>{state.error}</span>
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="flex items-center gap-2 px-5 py-3 border-t border-white/[0.04] bg-background">
        <span className="text-[13px] font-mono text-success select-none" aria-hidden="true">
          $
        </span>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => {
            setInput(e.target.value)
            if (historyIndex !== -1) setHistoryIndex(-1)
          }}
          onKeyDown={handleKeyDown}
          disabled={state.isRunning}
          placeholder={state.isRunning ? 'Running...' : placeholder}
          className="flex-1 bg-transparent font-mono text-[12px] text-foreground outline-none placeholder:text-muted-foreground disabled:opacity-50"
          data-testid="shell-input"
          aria-label="Shell command input"
        />
        {state.isRunning && (
          <button
            type="button"
            onClick={cancel}
            className="inline-flex items-center rounded-full px-2.5 py-0.5 text-[10px] font-medium bg-destructive/10 text-destructive hover:bg-destructive/20 transition-colors"
            data-testid="shell-cancel"
          >
            Cancel
          </button>
        )}
        {state.exitCode !== null && !hideExit && (
          <span
            data-testid="shell-exit-code"
            className={cn(
              'inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium',
              state.exitCode === 0
                ? 'bg-success/10 text-success'
                : 'bg-destructive/10 text-destructive',
            )}
          >
            exit {state.exitCode}
          </span>
        )}
      </div>
    </div>
  )
}
