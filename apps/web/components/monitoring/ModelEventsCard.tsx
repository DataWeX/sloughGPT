'use client'

import { useState, useEffect, memo } from 'react'
import { Card, CardContent } from '@sloughgpt/strui'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'
import { timeAgo } from '@/lib/time-ago'

interface ModelEventsCardProps {
  liveHealth: LiveHealthSnapshot | null
}

const EVENT_STYLES: Record<string, string> = {
  load: 'bg-success/15 text-success',
  unload: 'bg-muted text-muted-foreground',
  swap: 'bg-primary/15 text-primary',
  error: 'bg-destructive/15 text-destructive',
}

export const ModelEventsCard = memo(function ModelEventsCard({ liveHealth }: ModelEventsCardProps) {
  const events = liveHealth?.model_events ?? []
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 10_000)
    return () => clearInterval(id)
  }, [])

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Model events</span>
      <CardContent className="p-0 max-h-[220px] overflow-y-auto space-y-1.5" role="log" aria-live="polite" aria-label="Model event log">
        {events.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">No model events yet</p>
        ) : events.map((e, i) => (
          <div key={`${e.ts}-${i}`} className="border border-border/60 hover:bg-muted/50 transition-colors rounded-md p-2">
            <div className="flex items-center justify-between gap-2">
              <span
                className={`shrink-0 text-[9px] px-1.5 py-0.5 rounded font-medium uppercase ${
                  EVENT_STYLES[e.type] ?? 'bg-muted text-muted-foreground'
                }`}
              >
                {e.type}
              </span>
              <span className="truncate text-xs font-medium font-mono">{e.model}</span>
            </div>
            {e.detail && (
              <div className="text-[10px] text-muted-foreground mt-0.5 truncate" title={e.detail}>
                {e.detail}
              </div>
            )}
            <div className="text-[10px] opacity-60 mt-0.5 font-mono">{timeAgo(e.ts)}</div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
})
