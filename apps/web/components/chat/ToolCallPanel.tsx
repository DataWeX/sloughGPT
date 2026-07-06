'use client'

import { useState } from 'react'
import { cn } from '@/lib/cn'
import { Card } from '@sloughgpt/strui'
import type { ToolCallEvent } from '@/lib/stream-chat-response'

interface ToolCallPanelProps {
  events: ToolCallEvent[]
}

function ToolIcon({ name }: { name: string }) {
  const icons: Record<string, string> = {
    calculator: '\uD83E\uDEE6',
    current_time: '\uD83D\uDD52',
    web_search: '\uD83D\uDD0D',
    run_code: '\uD83D\uDCBB',
  }
  return <span className="mr-1.5 text-xs">{icons[name] || '\u2699\uFE0F'}</span>
}

function ToolCallCard({ event }: { event: ToolCallEvent }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <Card className={cn(
      'border px-3 py-2 text-xs transition-colors',
      event.status === 'executing' && 'border-blue-400/30 bg-blue-50/40 dark:bg-blue-950/20',
      event.status === 'success' && 'border-green-400/30 bg-green-50/40 dark:bg-green-950/20',
      event.status === 'error' && 'border-red-400/30 bg-red-50/40 dark:bg-red-950/20',
    )}>
      <button
        className="flex w-full items-center justify-between gap-2 text-left"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-1.5 min-w-0">
          <ToolIcon name={event.tool} />
          <span className="font-medium capitalize truncate">{event.tool}</span>
          {event.status === 'executing' && (
            <span className="inline-flex items-center gap-1 text-blue-600 dark:text-blue-400">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-pulse" />
              Running...
            </span>
          )}
          {event.status === 'success' && (
            <span className="text-green-600 dark:text-green-400">
              Done {event.duration_ms ? `(${(event.duration_ms / 1000).toFixed(1)}s)` : ''}
            </span>
          )}
          {event.status === 'error' && (
            <span className="text-red-600 dark:text-red-400">Failed</span>
          )}
        </div>
        {(event.output || event.error) && (
          <span className="text-muted-foreground shrink-0">{expanded ? '\u25B2' : '\u25BC'}</span>
        )}
      </button>
      {expanded && (event.output || event.error) && (
        <pre className="mt-2 overflow-x-auto rounded bg-muted/50 p-2 text-[11px] leading-relaxed whitespace-pre-wrap">
          {event.error ? (
            <span className="text-red-600 dark:text-red-400">{event.error}</span>
          ) : (
            event.output
          )}
        </pre>
      )}
    </Card>
  )
}

export function ToolCallPanel({ events }: ToolCallPanelProps) {
  if (!events.length) return null

  return (
    <div className="space-y-1.5 py-1">
      {events.map((ev, i) => (
        <ToolCallCard key={`${ev.tool}-${i}`} event={ev} />
      ))}
    </div>
  )
}
