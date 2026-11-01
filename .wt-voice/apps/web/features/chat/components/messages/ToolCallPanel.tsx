'use client'

import { useState, memo } from 'react'
import { cn } from '@sloughgpt/strui'
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
    code_execution: '\uD83D\uDCBB',
    file_read: '\uD83D\uDCC4',
    file_search: '\uD83D\uDD0D',
    knowledge_retrieval: '\uD83D\uDCD6',
    image_analysis: '\uD83D\uDDBC\uFE0F',
    data_analysis: '\uD83D\uDCCA',
    citation: '\uD83D\uDCD1',
  }
  return <span className="text-[10px]">{icons[name] || '\u2699\uFE0F'}</span>
}

const ToolCallCard = memo(function ToolCallCard({ event }: { event: ToolCallEvent }) {
  const [expanded, setExpanded] = useState(false)
  const hasOutput = !!(event.output || event.error)

  return (
    <div className={cn(
      'rounded-md border px-2 py-1 text-[10px] transition-colors',
      event.status === 'executing' && 'border-primary/30 bg-primary/5',
      event.status === 'success' && 'border-success/30 bg-success/5',
      event.status === 'error' && 'border-destructive/30 bg-destructive/5',
      !expanded && 'cursor-pointer hover:bg-muted/30',
    )}
      onClick={() => hasOutput && setExpanded(!expanded)}
      role={hasOutput ? 'button' : undefined}
      tabIndex={hasOutput ? 0 : undefined}
      onKeyDown={hasOutput ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          setExpanded(!expanded)
        }
      } : undefined}
      aria-expanded={hasOutput ? expanded : undefined}
    >
      <div className="flex items-center gap-1.5">
        <ToolIcon name={event.tool} />
        <span className="font-medium capitalize">{event.tool}</span>
        {event.status === 'executing' && (
          <span className="text-[9px] px-1 py-0.5 rounded bg-primary/10 text-primary animate-pulse">running</span>
        )}
        {event.status === 'success' && event.duration_ms != null && (
          <span className="text-[9px] text-success/70">{(event.duration_ms / 1000).toFixed(1)}s</span>
        )}
        {event.status === 'error' && (
          <span className="text-[9px] text-destructive">failed</span>
        )}
        {hasOutput && (
          <span className="ml-auto text-muted-foreground/50">{expanded ? '\u25B2' : '\u25BC'}</span>
        )}
      </div>
      {expanded && (event.output || event.error) && (
        <pre className="mt-1.5 overflow-x-auto rounded bg-muted/50 p-2 text-[10px] leading-relaxed whitespace-pre-wrap max-h-48 overflow-y-auto">
          {event.error ? (
            <span className="text-destructive">{event.error}</span>
          ) : (
            event.output
          )}
        </pre>
      )}
    </div>
  )
})

export const ToolCallPanel = memo(function ToolCallPanel({ events }: ToolCallPanelProps) {
  if (!events.length) return null

  const running = events.filter(e => e.status === 'executing')
  const completed = events.filter(e => e.status !== 'executing')

  return (
    <div className="space-y-1 py-0.5">
      {running.length > 0 && (
        <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
          <span>
            {running.map(e => e.tool).join(', ')}...
          </span>
        </div>
      )}
      {completed.map((ev, i) => (
        <ToolCallCard key={`${ev.tool}-${i}`} event={ev} />
      ))}
    </div>
  )
})
