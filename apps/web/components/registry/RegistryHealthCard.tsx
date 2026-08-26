'use client'

import { useMemo } from 'react'
import { cn, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import type { RegisteredModel, RegistryStats } from '@/lib/registry-controller'
import { timeAgo } from '@/lib/time-ago'

interface RegistryHealthCardProps {
  models: RegisteredModel[]
  stats: RegistryStats | null
}

function statusVariant(status: string): 'success' | 'error' | 'warning' | 'secondary' {
  if (status === 'loaded') return 'success'
  if (status === 'failed') return 'error'
  if (status === 'loading') return 'warning'
  return 'secondary'
}

export function RegistryHealthCard({ models, stats }: RegistryHealthCardProps) {
  const loaded = useMemo(() => models.filter(m => m.status === 'loaded'), [models])
  const failed = useMemo(() => models.filter(m => m.status === 'failed'), [models])
  const other = useMemo(() => models.filter(m => m.status !== 'loaded' && m.status !== 'failed'), [models])

  if (models.length === 0) return null

  return (
    <Card data-testid="registry-health">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle className="text-base">Health</CardTitle>
            {stats?.circuit_breaker_open && (
              <Badge label="Circuit Open" variant="error" size="sm" />
            )}
          </div>
          <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
            <span>{loaded.length} loaded</span>
            {failed.length > 0 && <span className="text-destructive">{failed.length} failed</span>}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-1.5">
          {models.map(m => {
            const metrics = m.metrics as Record<string, unknown> | undefined
            const failures = metrics?.failure_count as number | undefined
            const lastHealth = metrics?.last_health_check as string | undefined
            const requestCount = metrics?.request_count as number | undefined

            return (
              <div key={m.model_id} className="flex items-center justify-between text-[11px] py-1 border-b border-border/30 last:border-0">
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <span className={cn('h-1.5 w-1.5 rounded-full shrink-0', m.status === 'loaded' ? 'bg-success' :
                    m.status === 'failed' ? 'bg-destructive' :
                    'bg-muted-foreground/30')} />
                  <span className="font-medium truncate">{m.model_id}</span>
                  <Badge label={m.status} variant={statusVariant(m.status)} size="sm" />
                </div>
                <div className="flex items-center gap-3 shrink-0 text-muted-foreground">
                  {failures != null && failures > 0 && (
                    <span className="text-destructive">{failures} failures</span>
                  )}
                  {requestCount != null && (
                    <span>{requestCount} reqs</span>
                  )}
                  {lastHealth && (
                    <span>{timeAgo(lastHealth)}</span>
                  )}
                  {m.registered_at && !lastHealth && (
                    <span>{timeAgo(m.registered_at)}</span>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
