'use client'

import { useState, useEffect, useCallback, useMemo, memo } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import { XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'
import { trainingJobsController, type TrainingJob } from '@/lib/training-controller'
import { downloadJson } from '@/lib/download-utils'

interface Props {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}

const STATUS_COLORS: Record<string, string> = {
  completed: 'rgb(var(--success))',
  running: 'rgb(var(--warning))',
  failed: 'rgb(var(--destructive))',
  cancelled: 'rgb(var(--muted-foreground))',
}

export const TrainingAnalyticsCard = memo(function TrainingAnalyticsCard({ addToast }: Props) {
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loading, setLoading] = useState(true)

  const fetchJobs = useCallback(async () => {
    setLoading(true)
    try {
      const result = await trainingJobsController.list()
      setJobs(result ?? [])
    } catch {
      addToast('Could not fetch training data', 'error')
      setJobs([])
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      try {
        const result = await trainingJobsController.list()
        if (active) setJobs(result ?? [])
      } catch {
        if (active) {
          addToast('Could not fetch training data', 'error')
          setJobs([])
        }
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [addToast])

  const analytics = useMemo(() => {
    if (jobs.length === 0) return null

    // Status distribution
    const statusCounts = jobs.reduce((acc, j) => { acc[j.status] = (acc[j.status] || 0) + 1; return acc }, {} as Record<string, number>)
    const pieData = Object.entries(statusCounts).map(([name, value]) => ({ name, value }))

    // Loss over time (completed jobs with loss)
    const lossOverTime = jobs
      .filter(j => j.status === 'completed' && j.loss != null)
      .map(j => ({ name: j.name || j.id.slice(0, 6), loss: j.loss }))
      .slice(-10)

    // Duration distribution (completed jobs with elapsed_s)
    const durationBuckets = jobs
      .filter(j => j.status === 'completed' && j.elapsed_s != null)
      .map(j => ({ name: j.name || j.id.slice(0, 6), duration: Math.round(j.elapsed_s! / 60) }))

    // Method distribution
    const methodCounts = jobs.reduce((acc, j) => {
      const method = j.method || 'unknown'
      acc[method] = (acc[method] || 0) + 1
      return acc
    }, {} as Record<string, number>)
    const methodData = Object.entries(methodCounts).map(([name, count]) => ({ name, count }))

    // Summary stats
    const completed = jobs.filter(j => j.status === 'completed')
    const failed = jobs.filter(j => j.status === 'failed')
    const avgLoss = completed.length > 0
      ? completed.reduce((sum, j) => sum + (j.loss ?? 0), 0) / completed.length
      : null
    const avgDuration = completed.filter(j => j.elapsed_s != null).length > 0
      ? completed.filter(j => j.elapsed_s != null).reduce((sum, j) => sum + j.elapsed_s!, 0) / completed.filter(j => j.elapsed_s != null).length
      : null

    return { pieData, lossOverTime, durationBuckets, methodData, avgLoss, avgDuration, total: jobs.length, completed: completed.length, failed: failed.length }
  }, [jobs])

  const handleExport = useCallback(() => {
    if (!analytics) return
    downloadJson(analytics, `training-analytics-${new Date().toISOString().slice(0, 10)}.json`)
    addToast('Analytics exported', 'success')
  }, [analytics, addToast])

  if (loading) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Training Analytics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="rounded bg-muted/30 p-3">
                <div className="h-2 w-12 rounded bg-muted mb-2" />
                <div className="h-5 w-8 rounded bg-muted" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!analytics || jobs.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Training Analytics</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground text-center py-4">
            No training data yet. Complete training to see analytics.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Training Analytics</CardTitle>
          <Button size="sm" variant="ghost" onClick={handleExport}>Export</Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Summary stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="text-center">
            <p className="text-base font-semibold">{analytics.total}</p>
            <p className="text-[10px] text-muted-foreground">Total runs</p>
          </div>
          <div className="text-center">
            <p className="text-base font-semibold text-success">{analytics.completed}</p>
            <p className="text-[10px] text-muted-foreground">Completed</p>
          </div>
          <div className="text-center">
            <p className="text-base font-semibold text-destructive">{analytics.failed}</p>
            <p className="text-[10px] text-muted-foreground">Failed</p>
          </div>
          <div className="text-center">
            <p className="text-base font-semibold">{analytics.avgLoss != null ? analytics.avgLoss.toFixed(3) : '—'}</p>
            <p className="text-[10px] text-muted-foreground">Avg loss</p>
          </div>
        </div>

        {/* Status pie chart */}
        <div>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2">Status distribution</p>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={analytics.pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={30}
                  outerRadius={60}
                  dataKey="value"
                  label={({ name, value }) => `${name} (${value})`}
                  labelLine={false}
                >
                  {analytics.pieData.map((entry) => (
                    <Cell key={entry.name} fill={STATUS_COLORS[entry.name] ?? 'hsl(var(--muted-foreground))'} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Loss over time */}
        {analytics.lossOverTime.length > 1 && (
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2">Loss by job (recent)</p>
            <div className="h-40">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analytics.lossOverTime} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-border/30" />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'hsl(var(--background))', border: '1px solid hsl(var(--border))', borderRadius: '4px', fontSize: '11px' }}
                    formatter={(value: number) => [value.toFixed(4), 'Loss']}
                  />
                  <Bar dataKey="loss" fill="hsl(var(--primary))" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Duration over time */}
        {analytics.durationBuckets.length > 1 && (
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2">Duration by job (minutes)</p>
            <div className="h-40">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analytics.durationBuckets} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-border/30" />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'hsl(var(--background))', border: '1px solid hsl(var(--border))', borderRadius: '4px', fontSize: '11px' }}
                    formatter={(value: number) => [`${value} min`, 'Duration']}
                  />
                  <Bar dataKey="duration" fill="hsl(var(--accent))" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Method distribution */}
        {analytics.methodData.length > 0 && (
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2">Method distribution</p>
            <div className="flex flex-wrap gap-2">
              {analytics.methodData.map(m => (
                <div key={m.name} className="flex items-center gap-1.5 rounded bg-muted/30 px-2 py-1 text-xs">
                  <span className="font-medium">{m.name}</span>
                  <span className="text-muted-foreground">({m.count})</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Average duration */}
        {analytics.avgDuration != null && (
          <p className="text-[10px] text-muted-foreground">
            Average training duration: {Math.round(analytics.avgDuration / 60)}m {Math.round(analytics.avgDuration % 60)}s
          </p>
        )}
      </CardContent>
    </Card>
  )
})
