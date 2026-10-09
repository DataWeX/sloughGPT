'use client'

import { Card, CardHeader, CardTitle, CardContent, cn } from '@sloughgpt/strui'

interface Metric {
  metric: string
  value: number
  step: number
  timestamp: string
}

interface Param {
  param: string
  value: string
  timestamp: string
}

interface ExperimentStatsCardProps {
  metrics?: Metric[]
  params?: Param[]
  status?: { status: string; completed_at?: string } | null
}

export function ExperimentStatsCard({ metrics = [], params = [], status }: ExperimentStatsCardProps) {
  if (metrics.length === 0 && params.length === 0 && !status) return null

  const uniqueMetrics = [...new Set(metrics.map(m => m.metric))]
  const latestByMetric = uniqueMetrics.map(name => {
    const entries = metrics.filter(m => m.metric === name)
    return { name, latest: entries[entries.length - 1], count: entries.length }
  })

  return (
    <Card data-testid="experiment-stats">
      <CardHeader>
        <CardTitle className="text-base">Experiment Data</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {status && (
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Status:</span>
              <span className={cn(
                'text-[10px] px-1.5 py-0.5 rounded font-medium',
                status.status === 'completed' ? 'bg-success/15 text-success' :
                status.status === 'running' ? 'bg-primary/15 text-primary' :
                'bg-muted text-muted-foreground'
              )}>
                {status.status}
              </span>
              {status.completed_at && (
                <span className="text-[9px] text-muted-foreground">
                  Completed {new Date(status.completed_at).toLocaleString()}
                </span>
              )}
            </div>
          )}

          {latestByMetric.length > 0 && (
            <div>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-1.5">Metrics</div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {latestByMetric.map(m => (
                  <div key={m.name} className="p-2 rounded border border-border">
                    <div className="text-[9px] text-muted-foreground truncate">{m.name}</div>
                    <div className="text-sm font-semibold">{m.latest.value.toFixed(4)}</div>
                    <div className="text-[9px] text-muted-foreground">{m.count} entries</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {params.length > 0 && (
            <div>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-1.5">Parameters</div>
              <div className="space-y-0.5">
                {params.map((p, i) => (
                  <div key={i} className="flex items-center justify-between text-xs py-0.5">
                    <span className="text-muted-foreground">{p.param}</span>
                    <span className="font-mono">{p.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
