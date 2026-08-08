'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent } from '@sloughgpt/strui'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'

interface ServerErrorsCardProps {
  liveHealth: LiveHealthSnapshot | null
}

function timeAgo(ts: number): string {
  const s = Math.floor((Date.now() / 1000 - ts))
  if (s < 5) return 'just now'
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  return `${Math.floor(m / 60)}h ago`
}

export function ServerErrorsCard({ liveHealth }: ServerErrorsCardProps) {
  const errors = liveHealth?.recent_errors ?? []
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 10_000)
    return () => clearInterval(id)
  }, [])

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Server errors</span>
      <CardContent className="p-0 max-h-[220px] overflow-y-auto space-y-1.5">
        {errors.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">No errors recorded yet</p>
        ) : errors.map((e, i) => (
          <div key={`${e.ts}-${i}`} className="border border-destructive/20 hover:bg-destructive/5 transition-colors rounded-md p-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium truncate font-mono">
                {e.method} {e.path}
              </span>
              <span className="shrink-0 text-[9px] px-1.5 py-0.5 rounded font-medium bg-destructive/10 text-destructive tabular-nums">
                {e.status}
              </span>
            </div>
            <div className="text-[10px] text-muted-foreground mt-0.5 truncate" title={e.message}>
              {e.error_type && <span className="text-destructive/80">{e.error_type}: </span>}
              {e.message}
            </div>
            <div className="text-[10px] opacity-60 mt-0.5 font-mono">{timeAgo(e.ts)}</div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
