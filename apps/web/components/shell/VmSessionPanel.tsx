'use client'

import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { cn } from '@sloughgpt/strui'
import { useVmConsole, type VmConsolePhase } from '@/hooks/useVmConsole'

export interface VmSessionPanelProps {
  className?: string
  /** Role passed to POST /vm/session (default 'user'). Admin enables train commands. */
  role?: string
}

const PHASE_LABEL: Record<VmConsolePhase, string> = {
  idle: 'Idle',
  connecting: 'Connecting…',
  live: 'Console Live',
  complete: 'Process Halted',
  error: 'Error',
  closed: 'Closed',
}

const PHASE_DOT: Record<VmConsolePhase, string> = {
  idle: 'bg-muted-foreground',
  connecting: 'bg-warning animate-pulse',
  live: 'bg-success',
  complete: 'bg-muted-foreground',
  error: 'bg-destructive',
  closed: 'bg-muted-foreground',
}

/**
 * Interactive console for the project's own x86 VM (X86VirtualSystem).
 *
 * @example
 * ```tsx
 * <VmSessionPanel className="h-96" />
 * ```
 */
export function VmSessionPanel({ className, role }: VmSessionPanelProps) {
  const { phase, output, error, start, sendInput } = useVmConsole()
  const [input, setInput] = useState('')
  const startedRef = useRef(false)
  const scrollRef = useRef<HTMLPreElement>(null)
  const historyRef = useRef<string[]>([])
  const histIdxRef = useRef(-1)

  useEffect(() => {
    if (startedRef.current) return
    startedRef.current = true
    void start(role)
  }, [start, role])

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [output])

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    const text = input
    setInput('')
    histIdxRef.current = -1
    if (text.trim() !== '') {
      historyRef.current.push(text)
      if (historyRef.current.length > 50) historyRef.current.shift()
    }
    void sendInput(text.trim() === '' ? '\n' : `${text}\n`)
  }

  const onInputKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
    const hist = historyRef.current
    if (hist.length === 0) return
    e.preventDefault()
    let idx = histIdxRef.current
    if (e.key === 'ArrowUp') {
      idx = idx < 0 ? hist.length - 1 : Math.max(0, idx - 1)
    } else {
      if (idx < 0) return
      idx += 1
      if (idx >= hist.length) {
        histIdxRef.current = -1
        setInput('')
        return
      }
    }
    histIdxRef.current = idx
    setInput(hist[idx])
  }

  return (
    <div
      className={cn('flex flex-col rounded-lg border border-border bg-background', className)}
      data-testid="vm-session-panel"
    >
      {/* Status bar */}
      <div className="flex items-center gap-2 border-b border-border px-3 py-1.5">
        <div className={cn('h-2 w-2 rounded-full', PHASE_DOT[phase])} />
        <span className="text-xs text-muted-foreground" data-testid="vm-session-status">
          {PHASE_LABEL[phase]}
        </span>
        <span className="text-xs text-muted-foreground">own VM · X86VirtualSystem</span>
        <button
          type="button"
          onClick={() => void start(role)}
          className="ml-auto text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          Restart
        </button>
      </div>

      {/* Console output */}
      <pre
        ref={scrollRef}
        data-testid="vm-console-output"
        className="flex-1 overflow-y-auto whitespace-pre-wrap break-words bg-black px-3 py-2 font-mono text-[12px] leading-[1.6] text-[#c7c7cc]"
        aria-live="polite"
      >
        {output || ''}
      </pre>

      {error && (
        <div
          className="border-t border-border px-3 py-2 text-xs text-destructive"
          data-testid="vm-session-error"
        >
          {error}
        </div>
      )}

      {/* Input */}
      <form
        onSubmit={onSubmit}
        className="flex items-center gap-2 border-t border-border px-3 py-2"
      >
        <span className="font-mono text-[12px] text-[#28c840] select-none" aria-hidden="true">
          $
        </span>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onInputKeyDown}
          placeholder="type a command (help, ls, cat …)"
          aria-label="VM console input"
          data-testid="vm-session-input"
          className="flex-1 bg-transparent font-mono text-[12px] text-foreground outline-none placeholder:text-muted-foreground"
        />
        <button
          type="submit"
          className="rounded-md border border-border px-2 py-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          Send
        </button>
      </form>
    </div>
  )
}
