'use client'

import { memo } from 'react'
import { Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'

interface LatencyCardProps {
  liveHealth: LiveHealthSnapshot | null
}

export const LatencyCard = memo(function LatencyCard({ liveHealth }: LatencyCardProps) {
  const avg = liveHealth?.avg_latency_ms ?? 0
  const p95 = liveHealth?.p95_latency_ms ?? 0
  if (avg <= 0 && p95 <= 0) return null

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Latency</span>
      <CardContent className="p-0">
        <KpiGrid columns={2}>
          <StatCard label="Avg" value={avg > 0 ? avg.toFixed(0) + "ms" : <Skeleton className="h-5 w-12" />} numeric />
          <StatCard label="P95" value={p95 > 0 ? p95.toFixed(0) + "ms" : <Skeleton className="h-5 w-12" />} numeric />
        </KpiGrid>
      </CardContent>
    </Card>
  )
})
