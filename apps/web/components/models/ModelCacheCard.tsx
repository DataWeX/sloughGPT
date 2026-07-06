'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { KpiGrid, StatCard } from '@sloughgpt/strui'
import { modelDisplayName } from '@/lib/inference-display'
import { modelController } from '@/lib/model-controller'

interface ModelCacheCardProps {
  cacheUsage: { total_gb: number; model_count: number } | null
  health: any
  onRefresh: () => void
}

export default function ModelCacheCard({ cacheUsage, health, onRefresh }: ModelCacheCardProps) {
  const isLoaded = health !== null && health !== 'offline' && health.model_loaded
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Model Cache</CardTitle>
        <Button size="sm" variant="ghost" onClick={onRefresh}>
          <IconRefresh className="h-3.5 w-3.5" />
        </Button>
      </CardHeader>
      <CardContent>
        {cacheUsage ? (
          <KpiGrid columns={3}>
            <StatCard label="Cached Models" value={cacheUsage.model_count} />
            <StatCard label="Disk Usage" value={`${cacheUsage.total_gb.toFixed(1)} GB`} />
            <StatCard label="Loaded" value={isLoaded ? modelDisplayName(health.model_type) || 'Yes' : 'None'} />
          </KpiGrid>
        ) : (
          <p className="text-sm text-muted-foreground">Loading cache stats...</p>
        )}
      </CardContent>
    </Card>
  )
}
