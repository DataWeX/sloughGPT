'use client'

import { memo } from 'react'
import { Card, CardContent } from '@sloughgpt/strui'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'
import { formatTokens } from './TrafficCard'

interface ModelMetricsCardProps {
  liveHealth: LiveHealthSnapshot | null
}

export const ModelMetricsCard = memo(function ModelMetricsCard({ liveHealth }: ModelMetricsCardProps) {
  const metrics = liveHealth?.model_metrics ?? []
  if (metrics.length === 0) return null

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Model activity</span>
      <CardContent className="p-0 space-y-1.5">
        {metrics.map((m) => (
          <div key={m.model} className="border border-border/60 hover:bg-muted/50 transition-colors rounded-md p-2">
            <div className="text-sm font-medium truncate font-mono">{m.model}</div>
            <div className="text-[10px] text-muted-foreground mt-0.5">
              {m.count} request{m.count !== 1 ? 's' : ''} · {formatTokens(m.total_tokens)} tokens · {m.avg_tokens.toFixed(0)}/req
            </div>
            <div className="mt-1 flex flex-wrap gap-1.5">
              <span className="text-[9px] px-1.5 py-0.5 rounded font-medium bg-primary/10 text-primary">{m.tokens_per_sec.toFixed(1)} tok/s</span>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
})
