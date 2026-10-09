'use client'

import { memo } from 'react'
import { Card, CardContent, StatCard, KpiGrid, Skeleton, StatusDot } from '@sloughgpt/strui'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'
import type { SystemMetrics, DetailedHealth } from '@/lib/system-controller'
import { SectionLabel } from '@/components/composed/SectionLabel'

interface ResourceCardProps {
  liveHealth: LiveHealthSnapshot | null
  metrics: SystemMetrics | null
  detailed: DetailedHealth | null
  cpuThreshold: number
  memThreshold: number
  loaded: boolean
}

export const ResourceCard = memo(function ResourceCard({ liveHealth, metrics, detailed, cpuThreshold, memThreshold, loaded }: ResourceCardProps) {
  const cpu = liveHealth?.cpu_percent ?? metrics?.cpu_percent ?? null
  const mem = liveHealth?.memory_percent ?? metrics?.memory_percent ?? null

  const memUsedGB = metrics?.memory_used_gb ?? null
  const memTotalGB = metrics?.memory_total_gb ?? null
  const memAvailableGB = memUsedGB != null && memTotalGB != null
    ? Math.max(0, memTotalGB - memUsedGB)
    : detailed?.system?.memory_available_mb != null
      ? detailed.system.memory_available_mb / 1024
      : null

  return (
    <Card className="p-3">
      <SectionLabel>Resources</SectionLabel>
      <CardContent className="p-0">
        <KpiGrid columns={2}>
          <StatCard
            label="CPU"
            value={cpu != null ? cpu + '%' : <Skeleton className="h-5 w-10" />}
            numeric
            icon={<StatusDot tone={cpu != null && cpu > cpuThreshold ? 'warning' : 'success'} />}
          />
          <StatCard
            label="Memory"
            value={mem != null ? mem + '%' : <Skeleton className="h-5 w-10" />}
            numeric
            icon={<StatusDot tone={mem != null && mem > memThreshold ? 'warning' : 'success'} />}
          />
        </KpiGrid>
        {memUsedGB != null && memTotalGB != null && (
          <div className="mt-2 text-[10px] text-muted-foreground text-center">
            {memUsedGB.toFixed(1)} / {memTotalGB.toFixed(1)} GB
            {memAvailableGB != null ? ` (${memAvailableGB.toFixed(1)} GB free)` : ''}
          </div>
        )}
      </CardContent>
    </Card>
  )
})
