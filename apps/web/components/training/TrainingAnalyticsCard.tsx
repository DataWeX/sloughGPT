'use client'

import { useMemo, useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import { trainingJobsController } from '@/lib/training-controller'

export interface TrainingJobInfo {
  id: string
  name: string
  status: 'running' | 'completed' | 'failed' | 'pending'
  progress: number
  created_at: string
  method?: string
  loss?: number
  error?: string
}

export interface TrainingAnalyticsCardProps {
  addToast?: (msg: string, type?: 'success' | 'error' | 'info') => void
  onTrainMore?: () => void
}

export function TrainingAnalyticsCard({ addToast, onTrainMore }: TrainingAnalyticsCardProps) {
  const [jobs, setJobs] = useState<TrainingJobInfo[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    trainingJobsController.list()
      .then((data: TrainingJobInfo[]) => {
        if (!cancelled) {
          setJobs(data)
          setLoading(false)
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setLoading(false)
          addToast?.('Could not fetch training data', 'error')
        }
      })
    return () => { cancelled = true }
  }, [addToast])

  const stats = useMemo(() => {
    if (jobs.length === 0) return null

    const total = jobs.length
    const completed = jobs.filter(j => j.status === 'completed').length
    const failed = jobs.filter(j => j.status === 'failed').length
    const running = jobs.filter(j => j.status === 'running').length

    const losses = jobs.filter(j => j.loss != null)
    const avgLoss = losses.length > 0
      ? losses.reduce((s, j) => s + (j.loss ?? 0), 0) / losses.length
      : null

    const methodCounts: Record<string, number> = {}
    jobs.forEach(j => {
      if (j.method) {
        methodCounts[j.method] = (methodCounts[j.method] || 0) + 1
      }
    })

    return { total, completed, failed, running, avgLoss, methodCounts }
  }, [jobs])

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Training analytics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6">
            <p className="text-sm">Loading...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (jobs.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Training analytics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6">
            <p className="text-sm text-muted-foreground">
              No training data yet. Complete training to see analytics.
            </p>
            {onTrainMore && (
              <Button size="sm" variant="outline" className="mt-3" onClick={onTrainMore}>
                Train your first model
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Training analytics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <div>
            <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">Total runs</p>
            <p className="text-lg font-mono font-medium">{stats?.total}</p>
          </div>
          <div>
            <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">Completed</p>
            <p className="text-lg font-mono font-medium text-green-600">{stats?.completed}</p>
          </div>
          <div>
            <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">Failed</p>
            <p className="text-lg font-mono font-medium text-red-600">{stats?.failed}</p>
          </div>
          <div>
            <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">Avg loss</p>
            <p className="text-lg font-mono font-medium">
              {stats?.avgLoss != null ? stats.avgLoss.toFixed(4) : '-'}
            </p>
          </div>
        </div>

        {stats && Object.keys(stats.methodCounts).length > 0 && (
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground/60 uppercase tracking-wider">Method distribution</p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(stats.methodCounts).map(([method, count]) => (
                <div key={method} className="flex items-center gap-1.5">
                  <span className="text-xs font-mono">{method}</span>
                  <span className="text-xs text-muted-foreground">{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
