'use client'

import { useState, useRef, useEffect, type KeyboardEvent } from 'react'
import { cn } from '@sloughgpt/strui'
import { useShell, type ShellLine } from '@/hooks/useShell'

export interface ShellPanelProps {
  className?: string
  placeholder?: string
  maxVisibleLines?: number
}

/**
 * Terminal-like shell panel with command input and streaming output.
 *
 * @example
 * ```tsx
 * <ShellPanel className="h-96" />
 * ```
 */
export function ShellPanel({
  className,
  placeholder = 'Type a command...',
  maxVisibleLines = 500,
}: ShellPanelProps) {
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

    setHistory(prev => [...prev, cmd])
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
    <div className={cn(
      'flex flex-col rounded-lg border border-border bg-background',
      className,
    )}>
      {/* Output area */}
      <div
        ref={outputRef}
        className="flex-1 overflow-y-auto p-3 font-mono text-xs text-sm"
        data-testid="shell-output"
        role="log"
        aria-live="polite"
        aria-label="Shell output"
      >
        {visibleLines.length === 0 && !state.isRunning && placeholder && (
          <div className="text-muted-foreground">
            {placeholder}
          </div>
        )}
        {visibleLines.map((line: ShellLine) => (
          <div key={line.index} className="whitespace-pre-wrap break-all leading-relaxed">
            {line.text}
          </div>
        ))}
        {state.isRunning && (
          <div className="animate-pulse text-muted-foreground" data-testid="shell-running">...</div>
        )}
        {state.error && (
          <div className="mt-1 text-destructive" data-testid="shell-error">{state.error}</div>
        )}
      </div>

      {/* Input area */}
      <div className="flex items-center gap-2 border-t border-border px-3 py-2">
        <span className="text-xs text-muted-foreground" aria-hidden="true">$</span>
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
          className="flex-1 bg-transparent font-mono text-xs outline-none placeholder:text-muted-foreground disabled:opacity-50"
          data-testid="shell-input"
          aria-label="Shell command input"
        />
        {state.isRunning && (
          <button
            type="button"
            onClick={cancel}
            className="inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium bg-destructive/10 text-destructive hover:bg-destructive/20 transition-colors"
            data-testid="shell-cancel"
          >
            Cancel
          </button>
        )}
        {state.exitCode !== null && !hideExit && (
          <span
            data-testid="shell-exit-code"
            className={cn(
              'inline-flex items-center rounded-full px-1.5 py-0.5 text-[10px] font-medium',
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
