'use client'

import { useState, memo } from 'react'
import { cn, Card } from '@sloughgpt/strui'
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

const ToolCallCard = memo(function ToolCallCard({ event }: { event: ToolCallEvent }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <Card className={cn(
      'border px-3 py-2 text-xs transition-colors',
      event.status === 'executing' && 'border-primary/30 bg-primary/5',
      event.status === 'success' && 'border-success/30 bg-success/5',
      event.status === 'error' && 'border-destructive/30 bg-destructive/5',
    )}>
      <button
        type="button"
        className="flex w-full items-center justify-between gap-2 text-left"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
      >
        <div className="flex items-center gap-1.5 min-w-0">
          <ToolIcon name={event.tool} />
          <span className="font-medium capitalize truncate">{event.tool}</span>
          {event.status === 'executing' && (
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">
              Running...
            </span>
          )}
          {event.status === 'success' && (
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-success/10 text-success font-medium">
              Done {event.duration_ms ? `${(event.duration_ms / 1000).toFixed(1)}s` : ''}
            </span>
          )}
          {event.status === 'error' && (
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-destructive/10 text-destructive font-medium">Failed</span>
          )}
        </div>
        {(event.output || event.error) && (
          <span className="text-muted-foreground shrink-0">{expanded ? '\u25B2' : '\u25BC'}</span>
        )}
      </button>
      {expanded && (event.output || event.error) && (
        <pre className="mt-2 overflow-x-auto rounded bg-muted/50 p-2 text-[11px] leading-relaxed whitespace-pre-wrap">
          {event.error ? (
            <span className="text-destructive">{event.error}</span>
          ) : (
            event.output
          )}
        </pre>
      )}
    </Card>
  )
})

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
