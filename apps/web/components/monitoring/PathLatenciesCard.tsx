'use client'

import { memo } from 'react'
import { useMemo } from 'react'
import { Card, CardContent } from '@sloughgpt/strui'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'

interface PathLatenciesCardProps {
  liveHealth: LiveHealthSnapshot | null
}

export const PathLatenciesCard = memo(function PathLatenciesCard({ liveHealth }: PathLatenciesCardProps) {
  const latencies = liveHealth?.path_latencies ?? []
  if (latencies.length === 0) return null

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Endpoint latency</span>
      <CardContent className="p-0 space-y-1.5">
        {latencies.map((p) => (
          <div key={p.path} className="border border-border/60 hover:bg-muted/50 transition-colors rounded-md p-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium truncate font-mono">{p.path}</span>
              <span className="shrink-0 text-[9px] px-1.5 py-0.5 rounded font-medium bg-muted text-muted-foreground tabular-nums">
                ×{p.count}
              </span>
            </div>
            <div className="mt-1 flex flex-wrap gap-1.5">
              <span className="text-[9px] px-1.5 py-0.5 rounded font-medium bg-primary/10 text-primary">avg {p.avg_ms.toFixed(1)}ms</span>
              <span className="text-[9px] px-1.5 py-0.5 rounded font-medium bg-warning/10 text-warning">p95 {p.p95_ms.toFixed(1)}ms</span>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
})
