'use client'

import { memo } from 'react'
import { Card, CardContent } from '@sloughgpt/strui'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'
import { useTick } from '@/hooks/useTick'
import { timeAgo } from '@/lib/time-ago'

interface ServerErrorsCardProps {
  liveHealth: LiveHealthSnapshot | null
}

export const ServerErrorsCard = memo(function ServerErrorsCard({ liveHealth }: ServerErrorsCardProps) {
  const errors = liveHealth?.recent_errors ?? []
  useTick()

  return (
    <Card className="p-2.5">
      <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 block">Server errors</span>
      <CardContent className="p-0 max-h-[200px] overflow-y-auto space-y-1" role="log" aria-live="polite" aria-label="Server error log">
        {errors.length === 0 ? (
          <p className="text-[10px] text-muted-foreground/60 text-center py-3">No errors recorded yet</p>
        ) : errors.map((e, i) => (
          <div key={`${e.ts}-${i}`} className="border border-destructive/20 hover:bg-destructive/5 transition-colors rounded-md p-1.5">
            <div className="flex items-center justify-between gap-1.5">
              <span className="text-[10px] font-medium truncate font-mono">
                {e.method} {e.path}
              </span>
              <span className="shrink-0 text-[8px] px-1 py-0.5 rounded font-medium bg-destructive/10 text-destructive tabular-nums">
                {e.status}
              </span>
            </div>
            <div className="text-[9px] text-muted-foreground/60 mt-0.5 truncate" title={e.message}>
              {e.error_type && <span className="text-destructive/80">{e.error_type}: </span>}
              {e.message}
            </div>
            <div className="text-[9px] text-muted-foreground/40 mt-0.5 font-mono tabular-nums">{timeAgo(e.ts)}</div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
})
