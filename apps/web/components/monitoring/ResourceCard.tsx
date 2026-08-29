'use client'

import { memo } from 'react'
import { cn, Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'
import type { SystemMetrics, DetailedHealth } from '@/lib/system-controller'

interface ResourceCardProps {
  liveHealth: LiveHealthSnapshot | null
  metrics: SystemMetrics | null
  detailed: DetailedHealth | null
  cpuThreshold: number
  memThreshold: number
  loaded: boolean
}

function StatusDot({ value, threshold }: { value: number; threshold: number }) {
  return (
    <span className={cn('inline-block w-2 h-2 rounded-full', value > threshold ? 'bg-warning' : 'bg-success')} />
  )
}

export const ResourceCard = memo(function ResourceCard({ liveHealth, metrics, detailed, cpuThreshold, memThreshold, loaded }: ResourceCardProps) {
  const cpu = liveHealth?.cpu_percent ?? metrics?.cpu_percent ?? null
  const mem = liveHealth?.memory_percent ?? metrics?.memory_percent ?? null

  // Use consistent source: prefer /system/metrics for both Used and Total
  // so the math always adds up (Used + Available = Total)
  const memUsedGB = metrics?.memory_used_gb ?? null
  const memTotalGB = metrics?.memory_total_gb ?? null
  const memAvailableGB = memUsedGB != null && memTotalGB != null
    ? Math.max(0, memTotalGB - memUsedGB)
    : detailed?.system?.memory_available_mb != null
      ? detailed.system.memory_available_mb / 1024
      : null

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Resources</span>
      <CardContent className="p-0">
        <KpiGrid columns={2}>
          <StatCard
            label="CPU"
            value={cpu != null ? cpu + '%' : <Skeleton className="h-5 w-10" />}
            numeric
            icon={<StatusDot value={cpu ?? 0} threshold={cpuThreshold} />}
          />
          <StatCard
            label="Memory"
            value={mem != null ? mem + '%' : <Skeleton className="h-5 w-10" />}
            numeric
            icon={<StatusDot value={mem ?? 0} threshold={memThreshold} />}
          />
          <StatCard
            label="Used"
            value={memUsedGB != null ? memUsedGB.toFixed(1) + ' GB' : <Skeleton className="h-5 w-16" />}
            numeric
          />
          <StatCard
            label="Available"
            value={memAvailableGB != null ? memAvailableGB.toFixed(1) + ' GB' : <Skeleton className="h-5 w-16" />}
            numeric
          />
        </KpiGrid>
      </CardContent>
    </Card>
  )
})
