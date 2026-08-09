'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { experimentsController } from '@/lib/experiments-controller'

interface ExperimentDetailsCardProps {
  experimentId: string
}

interface ExperimentData {
  metrics: Array<{ metric: string; value: number; step: number; timestamp: string }>
  params: Array<{ param: string; value: string; timestamp: string }>
  status: { status: string; completed_at?: string } | null
}

export function ExperimentDetailsCard({ experimentId }: ExperimentDetailsCardProps) {
  const [data, setData] = useState<ExperimentData | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchData = async () => {
    setLoading(true)
    try {
      const result = await experimentsController.getExperimentData(experimentId)
      setData(result)
    } catch {
      // silently ignore
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [experimentId])

  if (loading) {
    return (
      <Card data-testid="experiment-details">
        <CardHeader><CardTitle className="text-base">Experiment Data</CardTitle></CardHeader>
        <CardContent><div className="h-20 animate-pulse bg-muted/50 rounded" /></CardContent>
      </Card>
    )
  }

  if (!data) return null

  const { metrics, params, status } = data
  const hasData = metrics.length > 0 || params.length > 0

  if (!hasData && !status) return null

  const uniqueMetrics = [...new Set(metrics.map(m => m.metric))]
  const latestByMetric: Record<string, { value: number; step: number }> = {}
  for (const m of metrics) {
    const existing = latestByMetric[m.metric]
    if (!existing || m.step >= existing.step) {
      latestByMetric[m.metric] = { value: m.value, step: m.step }
    }
  }

  const uniqueParams = [...new Set(params.map(p => p.param))]
  const latestByParam: Record<string, string> = {}
  for (const p of params) {
    latestByParam[p.param] = String(p.value)
  }

  return (
    <Card data-testid="experiment-details">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle className="text-base">Experiment Data</CardTitle>
            {status?.status && (
              <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                status.status === 'completed' ? 'bg-success/15 text-success' : 'bg-warning/15 text-warning'
              }`}>
                {status.status}
              </span>
            )}
          </div>
          <Button size="sm" variant="ghost" onClick={fetchData}>
            <IconRefresh className="h-3.5 w-3.5" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {uniqueMetrics.length > 0 && (
          <div className="mb-3">
            <div className="text-[10px] text-muted-foreground mb-1.5">Metrics</div>
            <div className="grid grid-cols-2 gap-2">
              {uniqueMetrics.map(m => (
                <div key={m} className="rounded-md bg-muted/30 p-2">
                  <div className="text-[10px] text-muted-foreground truncate">{m}</div>
                  <div className="text-sm font-mono font-medium">
                    {latestByMetric[m]?.value.toFixed(4)}
                  </div>
                  <div className="text-[9px] text-muted-foreground">
                    step {latestByMetric[m]?.step} · {metrics.filter(x => x.metric === m).length} entries
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {uniqueParams.length > 0 && (
          <div className="mb-3">
            <div className="text-[10px] text-muted-foreground mb-1.5">Parameters</div>
            <div className="space-y-1">
              {uniqueParams.map(p => (
                <div key={p} className="flex items-center justify-between text-[11px] py-0.5 border-b border-border/30 last:border-0">
                  <span className="text-muted-foreground">{p}</span>
                  <span className="font-mono">{latestByParam[p]}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {status?.completed_at && (
          <div className="text-[10px] text-muted-foreground">
            Completed {new Date(status.completed_at).toLocaleString()}
          </div>
        )}

        {!hasData && (
          <p className="text-sm text-muted-foreground text-center py-2">No metrics or parameters logged yet.</p>
        )}
      </CardContent>
    </Card>
  )
}
