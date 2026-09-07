'use client'

import { memo } from 'react'
import { cn, Card, CardContent, Button, StatCard, KpiGrid, IconRefresh } from '@sloughgpt/strui'
import { systemController } from '@/lib/system-controller'
import { useFetchCard } from '@/hooks/useFetchCard'

interface InferencePoolCardProps {
  onRefresh?: () => void
}

export const InferencePoolCard = memo(function InferencePoolCard({ onRefresh }: InferencePoolCardProps) {
  const { data: status, loading, error, refetch } = useFetchCard(
    () => systemController.getInferencePoolStatus(),
    [],
  )

  const handleRefresh = () => {
    refetch()
    onRefresh?.()
  }

  return (
    <Card data-testid="inference-pool">
      <div className="flex items-center justify-between border-b border-border/30 pb-2 pt-3 px-4">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Inference Pool</span>
        <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={handleRefresh} disabled={loading} aria-label="Refresh inference pool">
          <IconRefresh className={cn(loading && 'animate-spin', 'h-3 w-3')} />
        </Button>
      </div>
      <CardContent className="pt-3">
        {loading && !status ? (
          <KpiGrid>
            <StatCard label="Active" value="" loading />
            <StatCard label="Queued" value="" loading />
            <StatCard label="Avg Latency" value="" loading />
          </KpiGrid>
        ) : error ? (
          <p className="text-xs text-destructive text-center py-2">{error}</p>
        ) : status ? (
          <KpiGrid>
            <StatCard label="Active" value={String(status.active ?? 0)} />
            <StatCard label="Queued" value={String(status.queued ?? 0)} />
            <StatCard label="Avg Latency" value={status.avg_latency_ms != null ? `${Math.round(status.avg_latency_ms)}ms` : '—'} />
          </KpiGrid>
        ) : null}
      </CardContent>
    </Card>
  )
})
