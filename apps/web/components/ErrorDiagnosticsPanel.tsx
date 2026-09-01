'use client'

/**
 * ErrorDiagnosticsPanel — real-time error timeline with full context.
 *
 * Shows backend + frontend errors as they arrive, with:
 *   - Error level badges (error/critical/warning)
 *   - Source + timestamp
 *   - Correlation ID for tracing
 *   - Request context (method, path, status, duration)
 *   - Stack trace (expandable)
 *   - Quick actions: copy diagnostics, copy stack, open source
 *   - Filter by level and search by text
 *   - Group by fingerprint to collapse similar errors
 *
 * Used by DebugOverlay and the standalone diagnostics page.
 */

import { useState, useCallback, useMemo } from 'react'
import { cn, Button, IconX } from '@sloughgpt/strui'
import type { ErrorEvent } from '@/hooks/useErrorStream'

interface ErrorDiagnosticsPanelProps {
  errors: ErrorEvent[]
  onClear?: () => void
  className?: string
}

type LevelFilter = 'all' | 'error' | 'critical' | 'warning' | 'info'

interface GroupedError {
  fingerprint: string
  message: string
  level: ErrorEvent['level']
  count: number
  latest: ErrorEvent
  events: ErrorEvent[]
}

function levelBadge(level: ErrorEvent['level']): { label: string; color: string } {
  switch (level) {
    case 'critical': return { label: 'CRIT', color: 'bg-purple-500/20 text-purple-400 border-purple-500/30' }
    case 'error': return { label: 'ERR', color: 'bg-destructive/15 text-destructive border-destructive/30' }
    case 'warning': return { label: 'WRN', color: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30' }
    case 'info': return { label: 'INFO', color: 'bg-blue-500/15 text-blue-400 border-blue-500/30' }
    default: return { label: level, color: 'bg-muted text-muted-foreground' }
  }
}

function formatTime(ts: number): string {
  const d = new Date(ts)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function formatTimeAgo(ts: number): string {
  const diff = Date.now() - ts
  if (diff < 1000) return 'just now'
  if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`
  return `${Math.floor(diff / 3_600_000)}h ago`
}

function copyToClipboard(text: string) {
  navigator.clipboard.writeText(text).catch(() => {})
}

function groupErrors(errors: ErrorEvent[]): GroupedError[] {
  const groups = new Map<string, GroupedError>()
  for (const event of errors) {
    const fp = event.fingerprint || event.message.slice(0, 100)
    const existing = groups.get(fp)
    if (existing) {
      existing.count++
      existing.events.push(event)
      if (event.timestamp > existing.latest.timestamp) {
        existing.latest = event
      }
    } else {
      groups.set(fp, {
        fingerprint: fp,
        message: event.message,
        level: event.level,
        count: 1,
        latest: event,
        events: [event],
      })
    }
  }
  return Array.from(groups.values()).sort((a, b) => b.latest.timestamp - a.latest.timestamp)
}

function ErrorRow({ event, index }: { event: ErrorEvent; index: number }) {
  const [expanded, setExpanded] = useState(false)
  const badge = levelBadge(event.level)

  const diagnosticsText = useMemo(() => {
    const lines = [
      `Error: ${event.message}`,
      `Level: ${event.level}`,
      `Source: ${event.source}`,
      `Phase: ${event.phase}`,
      `Time: ${formatTime(event.timestamp)}`,
    ]
    if (event.correlationId) lines.push(`Correlation ID: ${event.correlationId}`)
    if (event.httpMethod && event.httpPath) {
      lines.push(`Request: ${event.httpMethod} ${event.httpPath}`)
    }
    if (event.httpStatus) lines.push(`Status: ${event.httpStatus}`)
    if (event.durationMs) lines.push(`Duration: ${event.durationMs}ms`)
    if (event.fingerprint) lines.push(`Fingerprint: ${event.fingerprint}`)
    if (event.url) lines.push(`URL: ${event.url}:${event.line ?? 0}:${event.col ?? 0}`)
    if (event.stack) lines.push(`Stack:\n${event.stack}`)
    if (event.context && Object.keys(event.context).length > 0) {
      lines.push(`Context: ${JSON.stringify(event.context, null, 2)}`)
    }
    return lines.join('\n')
  }, [event])

  const handleCopyAll = useCallback(() => copyToClipboard(diagnosticsText), [diagnosticsText])
  const handleCopyStack = useCallback(() => {
    if (event.stack) copyToClipboard(event.stack)
  }, [event.stack])

  return (
    <div
      className={cn(
        "group border-b border-border/20 last:border-0 transition-colors",
        expanded ? "bg-muted/30" : "hover:bg-muted/20",
      )}
    >
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left px-2 py-1.5 flex items-start gap-2"
      >
        <span className={cn("text-[9px] font-mono tabular-nums text-muted-foreground/40 shrink-0 w-5 text-right pt-0.5")}>
          {index + 1}
        </span>
        <span className={cn("shrink-0 text-[9px] font-semibold px-1 py-0.5 rounded border", badge.color)}>
          {badge.label}
        </span>
        <span className="flex-1 min-w-0">
          <span className="text-[10px] text-foreground/90 break-all leading-tight block">{event.message.slice(0, 200)}</span>
          <span className="text-[9px] text-muted-foreground/50 mt-0.5 block">
            {event.httpMethod && event.httpPath && (
              <span className="font-mono mr-1.5">{event.httpMethod} {event.httpPath}</span>
            )}
            {event.httpStatus && (
              <span className={cn(
                "font-mono mr-1.5",
                event.httpStatus >= 500 ? "text-destructive" : event.httpStatus >= 400 ? "text-yellow-400" : "text-muted-foreground/50",
              )}>{event.httpStatus}</span>
            )}
            {event.durationMs != null && (
              <span className="font-mono mr-1.5">{event.durationMs}ms</span>
            )}
            {event.source && <span className="mr-1.5">{event.source}</span>}
            <span>{formatTimeAgo(event.timestamp)}</span>
            {event.correlationId && <span className="ml-1.5 font-mono">[{event.correlationId}]</span>}
          </span>
        </span>
        <span className="shrink-0 text-muted-foreground/30 text-[9px] pt-0.5">
          {expanded ? '▾' : '▸'}
        </span>
      </button>

      {expanded && (
        <div className="px-2 pb-2 pl-9 space-y-1.5">
          <div className="flex items-center gap-1.5">
            <Button
              variant="ghost"
              size="sm"
              className="h-5 px-1.5 text-[9px] text-muted-foreground/60 hover:text-foreground"
              onClick={handleCopyAll}
            >
              Copy diagnostics
            </Button>
            {event.stack && (
              <Button
                variant="ghost"
                size="sm"
                className="h-5 px-1.5 text-[9px] text-muted-foreground/60 hover:text-foreground"
                onClick={handleCopyStack}
              >
                Copy stack
              </Button>
            )}
          </div>
          {event.url && (
            <div className="text-[9px] text-muted-foreground/50">
              at <span className="font-mono text-foreground/70">{event.url}:{event.line ?? 0}:{event.col ?? 0}</span>
            </div>
          )}
          {event.fingerprint && (
            <div className="text-[9px] text-muted-foreground/50">
              fingerprint: <span className="font-mono text-foreground/70">{event.fingerprint}</span>
              {event.count && event.count > 1 && (
                <span className="ml-1 text-yellow-400/70">×{event.count}</span>
              )}
            </div>
          )}
          {event.context && Object.keys(event.context).length > 0 && (
            <div className="text-[9px] font-mono text-muted-foreground/50 bg-muted/30 rounded p-1.5 max-h-24 overflow-auto">
              <pre className="whitespace-pre-wrap">{JSON.stringify(event.context, null, 2)}</pre>
            </div>
          )}
          {event.stack && (
            <div className="text-[9px] font-mono text-muted-foreground/40 bg-muted/30 rounded p-1.5 max-h-32 overflow-auto">
              <pre className="whitespace-pre-wrap">{event.stack}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function GroupedErrorRow({ group, index }: { group: GroupedError; index: number }) {
  const [expanded, setExpanded] = useState(false)
  const badge = levelBadge(group.level)

  const diagnosticsText = useMemo(() => {
    const lines = [
      `Error: ${group.message}`,
      `Level: ${group.level}`,
      `Count: ${group.count}`,
      `First: ${formatTime(group.events[0].timestamp)}`,
      `Latest: ${formatTime(group.latest.timestamp)}`,
    ]
    if (group.latest.correlationId) lines.push(`Correlation ID: ${group.latest.correlationId}`)
    if (group.latest.httpMethod && group.latest.httpPath) {
      lines.push(`Request: ${group.latest.httpMethod} ${group.latest.httpPath}`)
    }
    if (group.fingerprint) lines.push(`Fingerprint: ${group.fingerprint}`)
    if (group.latest.stack) lines.push(`Stack:\n${group.latest.stack}`)
    return lines.join('\n')
  }, [group])

  const handleCopyAll = useCallback(() => copyToClipboard(diagnosticsText), [diagnosticsText])

  return (
    <div
      className={cn(
        "group border-b border-border/20 last:border-0 transition-colors",
        expanded ? "bg-muted/30" : "hover:bg-muted/20",
      )}
    >
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left px-2 py-1.5 flex items-start gap-2"
      >
        <span className={cn("text-[9px] font-mono tabular-nums text-muted-foreground/40 shrink-0 w-5 text-right pt-0.5")}>
          {index + 1}
        </span>
        <span className={cn("shrink-0 text-[9px] font-semibold px-1 py-0.5 rounded border", badge.color)}>
          {badge.label}
        </span>
        <span className="flex-1 min-w-0">
          <span className="text-[10px] text-foreground/90 break-all leading-tight block">{group.message.slice(0, 200)}</span>
          <span className="text-[9px] text-muted-foreground/50 mt-0.5 block">
            {group.latest.httpMethod && group.latest.httpPath && (
              <span className="font-mono mr-1.5">{group.latest.httpMethod} {group.latest.httpPath}</span>
            )}
            {group.latest.httpStatus && (
              <span className={cn(
                "font-mono mr-1.5",
                group.latest.httpStatus >= 500 ? "text-destructive" : group.latest.httpStatus >= 400 ? "text-yellow-400" : "text-muted-foreground/50",
              )}>{group.latest.httpStatus}</span>
            )}
            {group.count > 1 && (
              <span className="text-yellow-400/70 mr-1.5">×{group.count}</span>
            )}
            <span>{formatTimeAgo(group.latest.timestamp)}</span>
          </span>
        </span>
        <span className="shrink-0 text-muted-foreground/30 text-[9px] pt-0.5">
          {expanded ? '▾' : '▸'}
        </span>
      </button>

      {expanded && (
        <div className="px-2 pb-2 pl-9 space-y-1.5">
          <div className="flex items-center gap-1.5">
            <Button
              variant="ghost"
              size="sm"
              className="h-5 px-1.5 text-[9px] text-muted-foreground/60 hover:text-foreground"
              onClick={handleCopyAll}
            >
              Copy diagnostics
            </Button>
          </div>
          <div className="text-[9px] text-muted-foreground/50">
            {group.count} occurrence{group.count !== 1 ? 's' : ''} — first {formatTimeAgo(group.events[0].timestamp)}, latest {formatTimeAgo(group.latest.timestamp)}
          </div>
          {group.fingerprint && (
            <div className="text-[9px] text-muted-foreground/50">
              fingerprint: <span className="font-mono text-foreground/70">{group.fingerprint}</span>
            </div>
          )}
          {group.latest.correlationId && (
            <div className="text-[9px] text-muted-foreground/50">
              latest correlation: <span className="font-mono text-foreground/70">{group.latest.correlationId}</span>
            </div>
          )}
          {group.latest.stack && (
            <div className="text-[9px] font-mono text-muted-foreground/40 bg-muted/30 rounded p-1.5 max-h-32 overflow-auto">
              <pre className="whitespace-pre-wrap">{group.latest.stack}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function ErrorDiagnosticsPanel({ errors, onClear, className }: ErrorDiagnosticsPanelProps) {
  const [levelFilter, setLevelFilter] = useState<LevelFilter>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [grouped, setGrouped] = useState(true)

  const filteredErrors = useMemo(() => {
    let result = errors
    if (levelFilter !== 'all') {
      result = result.filter(e => e.level === levelFilter)
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase()
      result = result.filter(e =>
        e.message.toLowerCase().includes(q) ||
        e.source.toLowerCase().includes(q) ||
        (e.correlationId && e.correlationId.toLowerCase().includes(q)) ||
        (e.httpPath && e.httpPath.toLowerCase().includes(q))
      )
    }
    return result
  }, [errors, levelFilter, searchQuery])

  const groupedErrors = useMemo(() => groupErrors(filteredErrors), [filteredErrors])

  return (
    <div className={cn("flex flex-col", className)}>
      <div className="flex items-center justify-between px-2 py-1 border-b border-border/30">
        <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
          Error Timeline
        </span>
        <div className="flex items-center gap-1">
          <span className="text-[9px] text-muted-foreground/40 tabular-nums">{filteredErrors.length}/{errors.length}</span>
          {onClear && errors.length > 0 && (
            <Button
              variant="ghost"
              size="icon"
              className="h-4 w-4 p-0 text-muted-foreground/40 hover:text-foreground"
              onClick={onClear}
              aria-label="Clear errors"
            >
              <IconX className="h-2.5 w-2.5" />
            </Button>
          )}
        </div>
      </div>
      <div className={cn(
        "px-2 py-1 border-b border-border/20 space-y-1",
        errors.length === 0 && "opacity-40 pointer-events-none",
      )}>
        <input
          type="text"
          placeholder="Filter errors..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-1.5 py-0.5 text-[9px] bg-muted/30 border border-border/30 rounded text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-border/50"
        />
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-0.5">
            {(['all', 'error', 'critical', 'warning', 'info'] as LevelFilter[]).map((level) => (
              <button
                key={level}
                type="button"
                onClick={() => setLevelFilter(level)}
                className={cn(
                  "px-1 py-0.5 text-[8px] rounded transition-colors",
                  levelFilter === level
                    ? "bg-primary/20 text-primary"
                    : "text-muted-foreground/50 hover:text-muted-foreground",
                )}
              >
                {level === 'all' ? 'ALL' : level === 'critical' ? 'CRIT' : level.toUpperCase()}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => setGrouped(!grouped)}
            className={cn(
              "px-1 py-0.5 text-[8px] rounded transition-colors",
              grouped
                ? "bg-primary/20 text-primary"
                : "text-muted-foreground/50 hover:text-muted-foreground",
            )}
          >
            {grouped ? 'GROUPED' : 'FLAT'}
          </button>
        </div>
      </div>
      <div className="max-h-48 overflow-y-auto">
        {filteredErrors.length === 0 ? (
          <div className="px-3 py-4 text-center text-[10px] text-muted-foreground/40">
            {errors.length === 0 ? 'No errors yet' : 'No matching errors'}
          </div>
        ) : grouped ? (
          groupedErrors.map((group, i) => (
            <GroupedErrorRow key={group.fingerprint} group={group} index={i} />
          ))
        ) : (
          filteredErrors.map((event, i) => (
            <ErrorRow key={event.id} event={event} index={i} />
          ))
        )}
      </div>
    </div>
  )
}
