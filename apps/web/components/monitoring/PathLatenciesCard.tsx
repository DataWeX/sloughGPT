'use client'

import { memo } from 'react'
import { Card, CardContent } from '@sloughgpt/strui'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'

interface PathLatenciesCardProps {
  liveHealth: LiveHealthSnapshot | null
}

export const PathLatenciesCard = memo(function PathLatenciesCard({ liveHealth }: PathLatenciesCardProps) {
  const latencies = liveHealth?.path_latencies ?? []
  if (latencies.length === 0) return null

  return (
    <Card className="p-2.5">
      <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 block">Endpoint latency</span>
      <CardContent className="p-0 space-y-1">
        {latencies.map((p) => (
          <div key={p.path} className="border border-border/40 hover:bg-muted/20 transition-colors rounded-md p-1.5">
            <div className="flex items-center justify-between gap-1.5">
              <span className="text-[10px] font-medium truncate font-mono">{p.path}</span>
              <span className="shrink-0 text-[8px] px-1 py-0.5 rounded font-medium bg-muted text-muted-foreground tabular-nums">
                ×{p.count}
              </span>
            </div>
            <div className="mt-0.5 flex flex-wrap gap-1">
              <span className="text-[8px] px-1 py-0.5 rounded font-medium bg-primary/10 text-primary">avg {p.avg_ms.toFixed(1)}ms</span>
              <span className="text-[8px] px-1 py-0.5 rounded font-medium bg-warning/10 text-warning">p95 {p.p95_ms.toFixed(1)}ms</span>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
})
