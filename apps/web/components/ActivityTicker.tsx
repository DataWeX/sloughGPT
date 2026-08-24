'use client'

import { useEffect, useState } from 'react'
import { useErrorStore } from '@/lib/error-store'
import { cn } from '@sloughgpt/strui'
import { timeAgo } from '@/lib/time-ago'

interface ActivityTickerProps {
  className?: string
  /** Show the full list on click */
  onExpand?: () => void
}

/**
 * Compact live ticker showing error activity.
 * Single line: dot + count + last error + time.
 * Expand into full list.
 */
export function ActivityTicker({ className, onExpand }: ActivityTickerProps) {
  const errors = useErrorStore(s => s.errors)
  const totalCount = useErrorStore(s => s.totalErrorCount)
  const [flash, setFlash] = useState(false)
  const [tick, setTick] = useState(0)

  // Flash on new errors
  useEffect(() => {
    if (errors.length === 0) return
    setFlash(true)
    const t = setTimeout(() => setFlash(false), 600)
    return () => clearTimeout(t)
  }, [errors.length])

  // Tick every 10s for relative timestamps
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 10_000)
    return () => clearInterval(id)
  }, [])

  const latest = errors[0]
  const errorCount = errors.reduce((sum, e) => sum + e.count, 0)

  if (errorCount === 0) {
    return (
      <div className={cn(
        'flex items-center gap-2 px-3 py-1.5 rounded-md text-xs',
        'bg-success/5 border border-success/20 text-success/70',
        className,
      )}>
        <span className="inline-block h-1.5 w-1.5 rounded-full bg-success shrink-0" />
        <span>No errors</span>
      </div>
    )
  }

  return (
    <button
      type="button"
      onClick={onExpand}
      className={cn(
        'flex items-center gap-2 px-3 py-1.5 rounded-md text-xs text-left w-full',
        'border transition-colors',
        flash
          ? 'bg-destructive/15 border-destructive/40 text-destructive'
          : 'bg-destructive/5 border-destructive/20 text-destructive/80 hover:bg-destructive/10',
        className,
      )}
      aria-label={`${errorCount} error${errorCount !== 1 ? 's' : ''}. ${latest?.title || ''}`}
    >
      <span className={cn(
        'inline-block h-1.5 w-1.5 rounded-full shrink-0',
        flash ? 'bg-destructive animate-pulse' : 'bg-destructive/60',
      )} />
      <span className="font-medium tabular-nums font-mono">{errorCount}</span>
      {latest && (
        <>
          <span className="truncate opacity-70">·</span>
          <span className="truncate flex-1 min-w-0" title={latest.message}>
            {latest.title !== 'Error' ? latest.title : latest.message.slice(0, 40)}
          </span>
          <span className="shrink-0 opacity-50 tabular-nums font-mono">{timeAgo(latest.timestamp)}</span>
        </>
      )}
    </button>
  )
}

/**
 * Expanded error list (used on monitoring page when user clicks ticker or sees full list).
 * Deduped, compact rows with dismiss.
 */
export function ErrorList({ className }: { className?: string }) {
  const errors = useErrorStore(s => s.errors)
  const dismissError = useErrorStore(s => s.dismissError)
  const clearErrors = useErrorStore(s => s.clearErrors)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 10_000)
    return () => clearInterval(id)
  }, [])

  if (errors.length === 0) return null

  const severityColor: Record<string, string> = {
    error: 'bg-destructive/10 border-destructive/30 text-destructive',
    warning: 'bg-warning/10 border-warning/30 text-warning',
    info: 'bg-muted border-border/60 text-muted-foreground',
  }

  return (
    <div className={cn('space-y-1', className)}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-muted-foreground font-mono">
          {errors.length} unique error{errors.length !== 1 ? 's' : ''}
        </span>
        <button type="button" onClick={clearErrors} className="text-xs text-muted-foreground hover:text-foreground">
          Clear all
        </button>
      </div>
      {errors.map(e => (
        <div key={e.id} className={`flex items-start gap-2 p-2 rounded border text-xs ${severityColor[e.severity] || severityColor.error}`}>
          <div className="flex-1 min-w-0">
            <div className="font-medium truncate flex items-center gap-1.5">
              {e.title}
              {e.count > 1 && (
                <span className="inline-flex items-center px-1 py-0.5 rounded-full bg-muted text-[9px] tabular-nums font-mono">
                  ×{e.count}
                </span>
              )}
            </div>
            <div className="truncate opacity-80">{e.message}</div>
            <div className="text-[10px] opacity-60 mt-0.5 font-mono">
              {timeAgo(e.timestamp)}
              {e.source && <> · {e.source}</>}
            </div>
          </div>
          <button
            type="button"
            onClick={() => dismissError(e.id)}
            className="shrink-0 opacity-50 hover:opacity-100 text-xs leading-none"
            aria-label="Dismiss"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  )
}
