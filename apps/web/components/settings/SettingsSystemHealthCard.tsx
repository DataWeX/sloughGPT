'use client'

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle, Button, Skeleton, StatCard, KpiGrid, IconRefresh, cn } from '@sloughgpt/strui'
import { formatUptime } from '@/lib/chat-utils'

interface DetailedHealth {
  status: string
  model_loaded?: boolean
  model_type?: string | null
  uptime_seconds?: number
  inference?: { inference_count?: number }
  gpu?: { backend: string; tier: string }
  versions?: { package?: string; api?: string; torch?: string; features?: Record<string, string> }
}

interface SystemMetrics {
  cpu_percent?: number
  memory_used_gb?: number
  memory_total_gb?: number
  memory_percent?: number
}

interface DiskUsage {
  used_gb?: number
  total_gb?: number
  percent?: number
}

interface SystemInfo {
  platform?: string
  platform_release?: string
  architecture?: string
  processor?: string
  cpu_count?: number
  platform_version?: string
}

interface SettingsSystemHealthCardProps {
  apiOk: boolean
  modelLoaded: boolean
  modelType: string | null
  detailed: DetailedHealth | null
  metrics: SystemMetrics | null
  disk: DiskUsage | null
  info: SystemInfo | null
  healthError: boolean
  onRefresh: () => void
}

export function SettingsSystemHealthCard({
  apiOk,
  modelLoaded,
  modelType,
  detailed,
  metrics,
  disk,
  info,
  healthError,
  onRefresh,
}: SettingsSystemHealthCardProps) {
  return (
    <Card>
      <CardHeader>
        <div>
          <CardTitle className="text-base">System health</CardTitle>
          <CardDescription>Backend status and resource usage</CardDescription>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {healthError && !detailed && !metrics ? (
          <div className="text-center py-6">
            <p className="text-sm text-destructive mb-3">Could not connect to service</p>
            <Button size="sm" variant="outline" onClick={onRefresh}>
              <IconRefresh className="h-3.5 w-3.5 mr-1.5" />
              Retry
            </Button>
          </div>
        ) : (
          <>
            <KpiGrid columns={4}>
              <StatCard
                label="API"
                value={<span className="font-mono">{apiOk ? 'Healthy' : 'Error'}</span>}
                icon={<span className={cn('inline-block w-2 h-2 rounded-full', apiOk ? 'bg-success' : 'bg-destructive')} />}
              />
              <StatCard
                label="Model"
                value={<span className="font-mono text-xs">{modelLoaded ? (modelType || 'Loaded') : 'None'}</span>}
                icon={<span className={cn('inline-block w-2 h-2 rounded-full', modelLoaded ? 'bg-success' : 'bg-muted-foreground/50')} />}
              />
              <StatCard
                label="Uptime"
                value={<span className="font-mono">{formatUptime(detailed?.uptime_seconds ?? 0)}</span>}
              />
              <StatCard
                label="Responses"
                value={<span className="font-mono">{String(detailed?.inference?.inference_count ?? 0)}</span>}
              />
            </KpiGrid>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <StatCard
                label="CPU"
                value={<span className="font-mono">{metrics ? `${metrics.cpu_percent}%` : <Skeleton className="h-5 w-10 inline-block" />}</span>}
                icon={<span className={cn('inline-block w-2 h-2 rounded-full', (metrics?.cpu_percent ?? 0) > 80 ? 'bg-warning' : 'bg-success')} />}
              />
              <StatCard
                label="Memory"
                value={<span className="font-mono">{metrics ? `${(metrics.memory_used_gb ?? 0).toFixed(1)} / ${(metrics.memory_total_gb ?? 0).toFixed(0)} GB` : <Skeleton className="h-5 w-20 inline-block" />}</span>}
                icon={<span className={cn('inline-block w-2 h-2 rounded-full', (metrics?.memory_percent ?? 0) > 80 ? 'bg-warning' : 'bg-success')} />}
              />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <StatCard
                label="Disk"
                value={<span className="font-mono">{disk ? `${(disk.used_gb ?? 0).toFixed(0)} / ${(disk.total_gb ?? 0).toFixed(0)} GB` : <Skeleton className="h-5 w-20 inline-block" />}</span>}
                icon={<span className={cn('inline-block w-2 h-2 rounded-full', (disk?.percent ?? 0) > 80 ? 'bg-warning' : 'bg-success')} />}
              />
              <StatCard
                label="GPU"
                value={<span className="font-mono text-xs">{detailed?.gpu ? `${detailed.gpu.backend.toUpperCase()} · ${detailed.gpu.tier}` : 'None'}</span>}
                icon={<span className={cn('inline-block w-2 h-2 rounded-full', detailed?.gpu ? 'bg-success' : 'bg-muted-foreground/50')} />}
              />
            </div>

            {info && (
              <div className="rounded-md bg-muted/50 px-3 py-2 text-xs text-muted-foreground font-mono flex flex-wrap gap-x-4 gap-y-1">
                <span>{info.platform} {info.platform_release}</span>
                <span>{info.architecture}</span>
                <span>{info.processor}</span>
                <span>{info.cpu_count} cores</span>
              </div>
            )}

            <div className="flex items-center justify-between">
              <Button variant="ghost" size="sm" className="text-xs" onClick={onRefresh}>
                Refresh health
              </Button>
            </div>
          </>
        )}
      </CardContent>
      <CardFooter className="justify-end">
        {detailed?.versions?.package && (
          <span className="text-[10px] font-mono px-1.5 py-0 h-5 text-muted-foreground border border-border/50 rounded text-xs mr-2">pkg {detailed.versions.package}</span>
        )}
        {detailed?.versions?.api && (
          <span className="text-[10px] font-mono px-1.5 py-0 h-5 text-muted-foreground border border-border/50 rounded text-xs">api {detailed.versions.api}</span>
        )}
      </CardFooter>
    </Card>
  )
}
