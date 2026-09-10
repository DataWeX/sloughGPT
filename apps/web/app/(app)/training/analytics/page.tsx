'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, Badge, Skeleton } from '@sloughgpt/strui'
import { settingsController } from '@/lib/settings-controller'
import { downloadJson } from '@/lib/download-utils'
import { BarChart3, TrendingUp, Target, Award, Tag, RefreshCw, Download } from 'lucide-react'

interface Analytics {
  total_runs: number
  avg_quality: number
  convergence_rate: number
  quality_trend: Array<{ run_id: string; quality: number; timestamp: string }>
  method_distribution: Record<string, number>
  model_distribution: Record<string, number>
  tag_cloud: Record<string, number>
  best_run: { run_id: string; model: string; quality_score: number; method: string } | null
  recent_runs: Array<{
    run_id: string; model: string; method: string; quality_score: number;
    converged: boolean; timestamp: string; tags: string[]
  }>
}

function StatCard({ label, value, icon: Icon, color }: { label: string; value: string | number; icon: React.ComponentType<{ className?: string }>; color: string }) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${color}`}>
            <Icon className="h-5 w-5 text-white" />
          </div>
          <div>
            <p className="text-sm text-muted-foreground">{label}</p>
            <p className="text-2xl font-bold">{value}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function BarChart({ data, maxBars = 8 }: { data: Array<{ label: string; value: number }>; maxBars?: number }) {
  const sorted = [...data].sort((a, b) => b.value - a.value).slice(0, maxBars)
  const maxVal = Math.max(...sorted.map(d => d.value), 1)
  const colors = ['bg-violet-500', 'bg-blue-500', 'bg-emerald-500', 'bg-amber-500', 'bg-rose-500', 'bg-cyan-500', 'bg-indigo-500', 'bg-pink-500']

  return (
    <div className="space-y-2">
      {sorted.map((d, i) => (
        <div key={d.label} className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground w-24 truncate" title={d.label}>{d.label}</span>
          <div className="flex-1 bg-muted rounded-full h-6 overflow-hidden">
            <div
              className={`h-full rounded-full ${colors[i % colors.length]} transition-all duration-500`}
              style={{ width: `${(d.value / maxVal) * 100}%` }}
            />
          </div>
          <span className="text-sm font-medium w-10 text-right">{d.value}</span>
        </div>
      ))}
      {sorted.length === 0 && <p className="text-sm text-muted-foreground">No data yet</p>}
    </div>
  )
}

function QualitySparkline({ data }: { data: Array<{ quality: number }> }) {
  if (data.length === 0) return <p className="text-sm text-muted-foreground">No trend data</p>
  const maxQ = Math.max(...data.map(d => d.quality), 1)
  const barW = Math.max(100 / data.length, 2)

  return (
    <div className="flex items-end gap-px h-20">
      {data.map((d, i) => (
        <div
          key={i}
          className="bg-violet-500 rounded-t-sm transition-all duration-300 hover:bg-violet-400"
          style={{ width: `${barW}%`, height: `${(d.quality / maxQ) * 100}%` }}
          title={`Run ${i + 1}: ${Math.round(d.quality * 100)}%`}
        />
      ))}
    </div>
  )
}

function TagCloud({ tags }: { tags: Record<string, number> }) {
  const entries = Object.entries(tags).sort((a, b) => b[1] - a[1])
  if (entries.length === 0) return <p className="text-sm text-muted-foreground">No tags yet</p>

  const maxCount = Math.max(...entries.map(e => e[1]), 1)
  return (
    <div className="flex flex-wrap gap-2">
      {entries.map(([tag, count]) => {
        const scale = 0.7 + (count / maxCount) * 0.6
        return (
          <Badge
            key={tag}
            className="bg-violet-100 text-violet-800 hover:bg-violet-200 transition-colors cursor-default"
            style={{ fontSize: `${scale}rem` }}
            title={`${count} run${count !== 1 ? 's' : ''}`}
          >
            {tag}
          </Badge>
        )
      })}
    </div>
  )
}

export default function TrainingAnalyticsPage() {
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await settingsController.getTrainingAnalytics()
      setAnalytics(data as unknown as Analytics)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load analytics')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const methodData = analytics ? Object.entries(analytics.method_distribution).map(([label, value]) => ({ label, value })) : []
  const modelData = analytics ? Object.entries(analytics.model_distribution).map(([label, value]) => ({ label, value })) : []

  return (
    <PageContainer title="Training Analytics">
      <AppRouteHeader left={<AppRouteHeaderLead title="Training Analytics" />} />

      {loading ? (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map(i => <Skeleton key={i} className="h-24 rounded-lg" />)}
          </div>
          <Skeleton className="h-48 rounded-lg" />
          <Skeleton className="h-48 rounded-lg" />
        </div>
      ) : error ? (
        <Card>
          <CardContent className="p-6 text-center">
            <p className="text-red-500 mb-4">{error}</p>
            <button onClick={load} className="text-sm text-violet-600 hover:underline">Retry</button>
          </CardContent>
        </Card>
      ) : analytics ? (
        <div className="space-y-6">
          <div className="flex justify-end gap-2">
            <button
              onClick={() => analytics && downloadJson(analytics, `training-analytics-${Date.now()}.json`)}
              className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <Download className="h-3 w-3" /> Export JSON
            </button>
            <button onClick={load} className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors">
              <RefreshCw className="h-3 w-3" /> Refresh
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <StatCard label="Total Runs" value={analytics.total_runs} icon={BarChart3} color="bg-violet-500" />
            <StatCard label="Avg Quality" value={`${Math.round(analytics.avg_quality * 100)}%`} icon={TrendingUp} color="bg-blue-500" />
            <StatCard label="Convergence Rate" value={`${Math.round(analytics.convergence_rate * 100)}%`} icon={Target} color="bg-emerald-500" />
            <StatCard label="Best Quality" value={analytics.best_run ? `${Math.round(analytics.best_run.quality_score * 100)}%` : '-'} icon={Award} color="bg-amber-500" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold mb-4">Quality Trend</h3>
                <QualitySparkline data={analytics.quality_trend} />
                <p className="text-xs text-muted-foreground mt-2">Last {analytics.quality_trend.length} runs</p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold mb-4">By Method</h3>
                <BarChart data={methodData} />
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold mb-4">By Model</h3>
                <BarChart data={modelData} />
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Tag className="h-4 w-4" /> Tags
                </h3>
                <TagCloud tags={analytics.tag_cloud} />
              </CardContent>
            </Card>
          </div>

          {analytics.best_run && (
            <Card>
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Award className="h-4 w-4 text-amber-500" /> Best Run
                </h3>
                <div className="flex items-center gap-4 text-sm">
                  <span className="font-mono text-muted-foreground">{analytics.best_run.run_id}</span>
                  <Badge className="bg-green-100 text-green-800">{Math.round(analytics.best_run.quality_score * 100)}%</Badge>
                  <span>{analytics.best_run.model}</span>
                  <span className="text-muted-foreground">{analytics.best_run.method}</span>
                </div>
              </CardContent>
            </Card>
          )}

          {analytics.recent_runs.length > 0 && (
            <Card>
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold mb-4">Recent Runs</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-left text-muted-foreground">
                        <th className="pb-2 font-medium">Run ID</th>
                        <th className="pb-2 font-medium">Model</th>
                        <th className="pb-2 font-medium">Method</th>
                        <th className="pb-2 font-medium">Quality</th>
                        <th className="pb-2 font-medium">Status</th>
                        <th className="pb-2 font-medium">Tags</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analytics.recent_runs.map(run => (
                        <tr key={run.run_id} className="border-b last:border-0">
                          <td className="py-2 font-mono text-xs">{run.run_id.slice(0, 8)}</td>
                          <td className="py-2">{run.model}</td>
                          <td className="py-2">{run.method}</td>
                          <td className="py-2">
                            <Badge className={run.quality_score >= 0.8 ? 'bg-green-100 text-green-800' : run.quality_score >= 0.6 ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800'}>
                              {Math.round(run.quality_score * 100)}%
                            </Badge>
                          </td>
                          <td className="py-2">
                            <Badge className={run.converged ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}>
                              {run.converged ? 'Converged' : 'Running'}
                            </Badge>
                          </td>
                          <td className="py-2">
                            <div className="flex gap-1 flex-wrap">
                              {run.tags?.map(t => (
                                <Badge key={t} className="bg-violet-100 text-violet-800 text-xs">{t}</Badge>
                              ))}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      ) : null}
    </PageContainer>
  )
}
