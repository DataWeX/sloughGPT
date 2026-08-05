'use client'

import { Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
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
    <span className={`inline-block w-2 h-2 rounded-full ${
      value < 0 ? 'bg-warning' : value > threshold ? 'bg-warning' : 'bg-success'
    }`} />
  )
}

export function ResourceCard({ liveHealth, metrics, detailed, cpuThreshold, memThreshold, loaded }: ResourceCardProps) {
  const cpu = liveHealth?.cpu_percent ?? metrics?.cpu_percent ?? null
  const mem = liveHealth?.memory_percent ?? metrics?.memory_percent ?? null

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Resources</span>
      <CardContent className="p-0">
        <KpiGrid columns={2}>
          <StatCard
            label="CPU"
            value={cpu != null ? <span className="font-mono">{cpu}%</span> : '...'}
            icon={<StatusDot value={cpu ?? -1} threshold={cpuThreshold} />}
          />
          <StatCard
            label="Memory"
            value={mem != null ? <span className="font-mono">{mem}%</span> : '...'}
            icon={<StatusDot value={mem ?? -1} threshold={memThreshold} />}
          />
          <StatCard
            label="Used"
            value={metrics ? <span className="font-mono">{metrics.memory_used_gb.toFixed(1)} GB</span> : '...'}
          />
          <StatCard
            label="Available"
            value={detailed?.system?.memory_available_mb ? <span className="font-mono">{(detailed.system.memory_available_mb / 1024).toFixed(1)} GB</span> : '...'}
          />
        </KpiGrid>
      </CardContent>
    </Card>
  )
}
