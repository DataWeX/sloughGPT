'use client'

import { useState, useRef, useEffect, type KeyboardEvent } from 'react'
import { cn } from '@sloughgpt/strui'
import { useShell, type ShellLine } from '@/hooks/useShell'

export interface TerminalPanelProps {
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
      'flex flex-col rounded-xl overflow-hidden',
      'border border-white/[0.08] shadow-2xl shadow-black/40',
      'bg-[#0a0a0a]',
      className,
    )}>
      {/* macOS title bar */}
      <div className="flex items-center h-11 px-4 bg-[#1c1c1e] border-b border-white/[0.06]">
        {/* Traffic lights */}
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-[#ff5f57] shadow-[inset_0_-1px_1px_rgba(0,0,0,0.15)] ring-1 ring-black/10" />
          <span className="w-3 h-3 rounded-full bg-[#febc2e] shadow-[inset_0_-1px_1px_rgba(0,0,0,0.15)] ring-1 ring-black/10" />
          <span className="w-3 h-3 rounded-full bg-[#28c840] shadow-[inset_0_-1px_1px_rgba(0,0,0,0.15)] ring-1 ring-black/10" />
        </div>
        {/* Title */}
        <div className="flex-1 flex items-center justify-center">
          <span className="text-[11px] font-medium text-[#8e8e93] select-none">dait</span>
        </div>
        {/* Status indicator */}
        <div className="flex items-center gap-1.5">
          <span className={cn(
            'w-1.5 h-1.5 rounded-full',
            state.isRunning ? 'bg-[#febc2e] animate-pulse' :
            state.exitCode !== null && state.exitCode !== 0 ? 'bg-[#ff5f57]' :
            'bg-[#28c840]',
          )} />
        </div>
      </div>

      {/* Output area */}
      <div
        ref={outputRef}
        className="flex-1 overflow-y-auto px-5 py-4 font-mono text-[12px] leading-[1.7] text-[#c7c7cc]"
        data-testid="shell-output"
        role="log"
        aria-live="polite"
        aria-label="Shell output"
      >
        {visibleLines.length === 0 && !state.isRunning && placeholder && (
          <div className="text-[#636366] italic">
            {placeholder}
          </div>
        )}
        {visibleLines.map((line: ShellLine) => (
          <div
            key={line.index}
            className={cn(
              'whitespace-pre-wrap break-all',
              line.text.startsWith('Error') || line.text.startsWith('error')
                ? 'text-[#ff5f57]'
                : 'text-[#c7c7cc]',
            )}
          >
            {line.text}
          </div>
        ))}
        {state.isRunning && (
          <div className="flex items-center gap-2 text-[#febc2e]" data-testid="shell-running">
            <span className="w-1.5 h-1.5 rounded-full bg-[#febc2e] animate-pulse" />
            <span className="text-[10px]">executing...</span>
          </div>
        )}
        {state.error && (
          <div className="mt-2 flex items-start gap-2 text-[#ff5f57] bg-[#ff5f57]/[0.08] rounded-md px-3 py-2" data-testid="shell-error">
            <span className="shrink-0 text-[10px] font-bold">!</span>
            <span>{state.error}</span>
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="flex items-center gap-2 px-5 py-3 border-t border-white/[0.04] bg-[#111111]">
        <span className="text-[13px] font-mono text-[#28c840] select-none" aria-hidden="true">$</span>
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
          className="flex-1 bg-transparent font-mono text-[12px] text-[#e5e5ea] outline-none placeholder:text-[#48484a] disabled:opacity-50"
          data-testid="shell-input"
          aria-label="Shell command input"
        />
        {state.isRunning && (
          <button
            type="button"
            onClick={cancel}
            className="inline-flex items-center rounded-full px-2.5 py-0.5 text-[10px] font-medium bg-[#ff5f57]/10 text-[#ff5f57] hover:bg-[#ff5f57]/20 transition-colors"
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
                ? 'bg-[#28c840]/10 text-[#28c840]'
                : 'bg-[#ff5f57]/10 text-[#ff5f57]',
            )}
          >
            exit {state.exitCode}
          </span>
        )}
      </div>
    </div>
  )
}
