'use client'

import { cn, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { KpiGrid, StatCard } from '@sloughgpt/strui'
import { modelDisplayName } from '@/lib/inference-display'
import { modelController, type HealthStatus } from '@/lib/model-controller'

interface ModelCacheCardProps {
  cacheUsage: { total_gb: number; model_count: number } | null
  health: HealthStatus | null
  onRefresh: () => void
}

export default function ModelCacheCard({ cacheUsage, health, onRefresh }: ModelCacheCardProps) {
  const isLoaded = health !== null && health.model_loaded
  const maxCacheGb = 10
  const usagePercent = cacheUsage ? Math.min((cacheUsage.total_gb / maxCacheGb) * 100, 100) : 0
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Model Cache</CardTitle>
        <Button size="sm" variant="ghost" onClick={onRefresh} aria-label="Refresh cache">
          <IconRefresh className="h-3.5 w-3.5" />
        </Button>
      </CardHeader>
      <CardContent>
        {cacheUsage ? (
          <>
            <KpiGrid columns={3}>
              <StatCard label="Cached Models" value={cacheUsage.model_count} />
              <StatCard label="Disk Usage" value={`${cacheUsage.total_gb.toFixed(1)} GB`} />
              <StatCard label="Loaded" value={isLoaded ? modelDisplayName(health.model_type) || 'Yes' : 'None'} />
            </KpiGrid>
            <div className="mt-3">
              <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
                <span>Cache usage</span>
                <span>{cacheUsage.total_gb.toFixed(1)} / {maxCacheGb} GB</span>
              </div>
              <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                <div
                  className={cn('h-full transition-all duration-300 rounded-full', usagePercent > 80 ? 'bg-warning' : 'bg-primary')}
                  style={{ width: `${usagePercent}%` }}
                />
              </div>
            </div>
          </>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-3">
              {[1, 2, 3].map(i => (
                <div key={i} className="space-y-1.5">
                  <Skeleton className="h-3 w-16" />
                  <Skeleton className="h-5 w-12" />
                </div>
              ))}
            </div>
            <Skeleton className="h-1.5 w-full rounded-full" />
          </div>
        )}
      </CardContent>
    </Card>
  )
}
