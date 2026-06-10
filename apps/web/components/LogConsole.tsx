'use client'

import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/cn'

export type LogLevel = 'all' | 'info' | 'warn' | 'error'

export interface LogLine {
  id: string
  ts: string
  level: Exclude<LogLevel, 'all'>
  message: string
}

function levelTag(level: Exclude<LogLevel, 'all'>) {
  switch (level) {
    case 'error':
      return 'ERR'
    case 'warn':
      return 'WRN'
    default:
      return 'INF'
  }
}

function levelStyle(level: Exclude<LogLevel, 'all'>) {
  switch (level) {
    case 'error':
      return 'text-destructive'
    case 'warn':
      return 'text-warning'
    default:
      return 'text-chart-3'
  }
}

function levelDot(level: Exclude<LogLevel, 'all'>) {
  switch (level) {
    case 'error':
      return 'bg-destructive shadow-[0_0_0.375rem_color-mix(in_srgb,rgb(var(--destructive))_55%,transparent)]'
    case 'warn':
      return 'bg-warning shadow-[0_0_0.375rem_color-mix(in_srgb,rgb(var(--warning))_50%,transparent)]'
    default:
      return 'bg-chart-3 shadow-[0_0_0.375rem_color-mix(in_srgb,rgb(var(--chart-3))_45%,transparent)]'
  }
}

const TAB_ITEMS: { value: LogLevel; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'info', label: 'Info' },
  { value: 'warn', label: 'Warn' },
  { value: 'error', label: 'Error' },
]

export interface LogConsoleProps {
  /** Bump after an external event (e.g. monitoring poll) to append a status line. */
  tick?: number
  className?: string
}

export function LogConsole({ tick = 0, className }: LogConsoleProps) {
  const baseId = useId()
  const [lines, setLines] = useState<LogLine[]>(() => seedLines())
  const [filter, setFilter] = useState<LogLevel>('all')
  /** One append per distinct tick (avoids duplicate lines when React Strict Mode re-runs effects in dev). */
  const lastPollTickLogged = useRef<number | null>(null)

  const append = useCallback(
    (level: Exclude<LogLevel, 'all'>, message: string) => {
      const ts = new Date().toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
      setLines((prev) => {
        const next: LogLine[] = [
          ...prev,
          {
            id: `${baseId}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            ts,
            level,
            message,
          },
        ]
        return next.slice(-200)
      })
    },
    [baseId],
  )

  useEffect(() => {
    if (tick <= 0) return
    if (lastPollTickLogged.current === tick) return
    lastPollTickLogged.current = tick
    append('info', `[poll] metrics.tick=${tick} route=GET /info status=ok`)
  }, [tick, append])

  const filtered = useMemo(() => {
    if (filter === 'all') return lines
    return lines.filter((l) => l.level === filter)
  }, [lines, filter])

  const clear = () => {
    setLines(seedLines())
  }

  return (
    <div
      className={cn(
        'overflow-hidden rounded-lg border border-border/90 bg-card/90 shadow-sm ring-1 ring-border/50 backdrop-blur-sm',
        'shadow-primary',
        className,
      )}
    >
      <div className="flex flex-col border-b border-border/80 bg-gradient-to-r from-secondary/40 via-card/60 to-secondary/30">
        <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 font-mono">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <span
              className="h-2 w-2 shrink-0 rounded-full bg-success shadow-[0_0_0.5rem_color-mix(in_srgb,rgb(var(--success))_60%,transparent)]"
              aria-hidden
            />
            <span className="text-[0.625rem] font-semibold uppercase tracking-[0.22em] text-primary">SYS.LOG</span>
            <span className="hidden text-[0.625rem] text-muted-foreground sm:inline">
              {'//'} telemetry · stdout
              {tick > 0 ? ' · live' : ' · idle'}
            </span>
          </div>
          <span className="text-[0.5625rem] uppercase tracking-wider text-muted-foreground/90">man.web</span>
        </div>

        <div className="flex flex-col border-t border-border/50 sm:flex-row sm:items-stretch sm:justify-between">
          <div
            className="flex flex-wrap items-center gap-0 p-1 sm:flex-nowrap"
            role="tablist"
            aria-label="Log level filter"
          >
            {TAB_ITEMS.map(({ value, label }) => {
              const active = filter === value
              return (
                <Button
                  key={value}
                  variant="ghost"
                  size="sm"
                  role="tab"
                  aria-selected={active}
                  onClick={() => setFilter(value)}
                  className={cn(
                    'h-7 min-w-[3rem] rounded-md px-2.5 py-0 font-mono text-[0.625rem] font-semibold uppercase tracking-wider',
                    active
                      ? 'bg-primary/10 text-primary shadow-[inset_0_-0.125rem_0_0_rgb(var(--primary))]'
                      : '',
                  )}
                >
                  {label}
                </Button>
              )
            })}
          </div>
<div className="flex items-center justify-between gap-2 border-t border-border/50 px-2 py-1.5 sm:border-t-0 sm:border-l sm:border-border/50 sm:py-0">
            <span className="font-mono text-[0.625rem] tabular-nums text-muted-foreground" aria-live="polite">
              buf:{filtered.length}
            </span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-7 rounded-md px-2 font-mono text-[0.625rem] uppercase tracking-wide"
              onClick={clear}
              aria-label="Clear log"
            >
              Clear
            </Button>
          </div>
        </div>
      </div>

      <div
        className="sl-log-feed max-h-52 overflow-y-auto overscroll-contain px-3 py-2.5 font-mono text-[0.6875rem] leading-relaxed"
        role="log"
        aria-live="polite"
        aria-relevant="additions"
      >
        {filtered.length === 0 ? (
          <p className="py-8 text-center font-mono text-[0.625rem] uppercase tracking-wider text-muted-foreground">
            ∅ no frames for filter
          </p>
        ) : (
          <ul className="space-y-1.5">
            {filtered.map((line) => (
              <li key={line.id} className="flex gap-2 break-all border-l border-transparent pl-1 hover:border-primary/25">
                <span className="shrink-0 tabular-nums text-muted-foreground/90">[{line.ts}]</span>
                <span className={cn('w-7 shrink-0 font-semibold tabular-nums text-[0.5625rem]', levelStyle(line.level))}>
                  {levelTag(line.level)}
                </span>
                <span className={cn('mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full', levelDot(line.level))} aria-hidden />
                <span className={cn('min-w-0 flex-1', levelStyle(line.level))}>{line.message}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

function seedLines(): LogLine[] {
  const ts = () =>
    new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
  return [
    {
      id: 's1',
      ts: ts(),
      level: 'info',
      message: '[boot] console online · NEXT_PUBLIC_API_URL → inference plane',
    },
    {
      id: 's2',
      ts: ts(),
      level: 'info',
      message: '[auth] session provider hydrated · theme sync active',
    },
    {
      id: 's3',
      ts: ts(),
      level: 'warn',
      message: '[net] if API unreachable, status cards show disconnected — check CORS + /health',
    },
  ]
}
