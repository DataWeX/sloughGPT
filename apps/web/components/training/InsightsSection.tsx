'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription, Badge, Button, Skeleton, IconDownload, IconRefresh } from '@sloughgpt/strui'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend, Area, ComposedChart, Bar, BarChart,
} from 'recharts'
import { useTrainingInsights } from '@/hooks/useTrainingInsights'
import { useTrainingTrends } from '@/hooks/useTrainingTrends'
import { useTrainingAnalytics } from '@/hooks/useTrainingAnalytics'
import { downloadJson } from '@/lib/download-utils'
import { TrendIcon, TrainingChartTooltip, QualityBadge, RunStatusBadge, formatRunDate } from './training-shared'
import { Brain, Lightbulb, Target, Sparkles, TrendingUp, BarChart3, Award, Tag } from 'lucide-react'

function DistributionBars({ data, maxBars = 8 }: { data: Array<{ label: string; value: number }>; maxBars?: number }) {
  const sorted = [...data].sort((a, b) => b.value - a.value).slice(0, maxBars)
  const maxVal = Math.max(...sorted.map(d => d.value), 1)
  const colors = ['bg-violet-500', 'bg-blue-500', 'bg-emerald-500', 'bg-amber-500', 'bg-rose-500', 'bg-cyan-500', 'bg-indigo-500', 'bg-pink-500']
  return (
    <div className="space-y-2">
      {sorted.map((d, i) => (
        <div key={d.label} className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground w-24 truncate" title={d.label}>{d.label}</span>
          <div className="flex-1 bg-muted rounded-full h-6 overflow-hidden">
            <div className={`h-full rounded-full ${colors[i % colors.length]}`} style={{ width: `${(d.value / maxVal) * 100}%` }} />
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
  return (
    <div className="flex items-end gap-px h-20">
      {data.map((d, i) => (
        <div
          key={i}
          className="bg-violet-500 rounded-t-sm hover:bg-violet-400"
          style={{ width: `${Math.max(100 / data.length, 2)}%`, height: `${(d.quality / maxQ) * 100}%` }}
          title={`Run ${i + 1}: ${Math.round(d.quality * 100)}%`}
        />
      ))}
    </div>
  )
}

export function InsightsSection() {
  const insights = useTrainingInsights()
  const trends = useTrainingTrends()
  const analytics = useTrainingAnalytics()
  const [refreshing, setRefreshing] = useState(false)

  const loading = insights.loading || trends.loading || analytics.loading
  const hasInsights = insights.insights && (insights.insights.total_runs ?? 0) > 0
  const hasMessage = insights.insights?.message

  const refreshAll = async () => {
    setRefreshing(true)
    try {
      await Promise.all([insights.loadData(), trends.loadTrends(), analytics.load()])
    } finally {
      setRefreshing(false)
    }
  }

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  if (hasMessage && !hasInsights) {
    return (
      <Card>
        <CardContent className="py-16 text-center">
          <Brain className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <p className="text-lg font-medium mb-2">No Training History Yet</p>
          <p className="text-sm text-muted-foreground max-w-md mx-auto">{insights.insights?.message}</p>
        </CardContent>
      </Card>
    )
  }

  const methodData = analytics.analytics ? Object.entries(analytics.analytics.method_distribution).map(([label, value]) => ({ label, value })) : []
  const modelData = analytics.analytics ? Object.entries(analytics.analytics.model_distribution).map(([label, value]) => ({ label, value })) : []

  return (
    <div className="space-y-6">
      <div className="flex justify-end gap-2">
        <Button variant="outline" size="sm" onClick={refreshAll} disabled={refreshing} leftIcon={<IconRefresh className="h-3 w-3" />}>
          Refresh
        </Button>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card><CardContent className="pt-4"><div className="flex items-center gap-2 mb-1"><Target className="h-4 w-4 text-muted-foreground" /><p className="text-xs text-muted-foreground">Total Runs</p></div><p className="text-3xl font-bold">{insights.insights?.total_runs ?? trends.summary?.total_runs ?? 0}</p></CardContent></Card>
        <Card><CardContent className="pt-4"><div className="flex items-center gap-2 mb-1"><Sparkles className="h-4 w-4 text-muted-foreground" /><p className="text-xs text-muted-foreground">Avg Quality</p></div><p className="text-3xl font-bold">{(insights.insights?.avg_quality ?? trends.summary?.avg_quality ?? 0).toFixed(3)}</p></CardContent></Card>
        <Card><CardContent className="pt-4"><div className="flex items-center gap-2 mb-1"><TrendingUp className="h-4 w-4 text-green-500" /><p className="text-xs text-muted-foreground">Best Quality</p></div><p className="text-3xl font-bold text-green-500">{(insights.insights?.best_quality ?? trends.summary?.best_quality ?? 0).toFixed(3)}</p></CardContent></Card>
        <Card><CardContent className="pt-4"><div className="flex items-center gap-2 mb-1"><TrendIcon trend={insights.insights?.trend ?? trends.summary?.trend ?? 'stable'} /><p className="text-xs text-muted-foreground">Trend</p></div><p className="text-3xl font-bold capitalize">{insights.insights?.trend ?? trends.summary?.trend ?? '—'}</p></CardContent></Card>
      </div>

      {/* Quality over time */}
      {trends.chartPoints.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Quality Over Time</CardTitle>
            <CardDescription>Training quality score per run (0-1, higher is better)</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <ComposedChart data={trends.chartPoints}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip content={<TrainingChartTooltip />} />
                <Legend />
                <Area type="monotone" dataKey="quality" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.1} name="Quality" />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Loss over time */}
      {trends.chartPoints.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Loss Over Time</CardTitle>
            <CardDescription>Final training loss per run (lower is better)</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={260}>
              <ComposedChart data={trends.chartPoints}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip content={<TrainingChartTooltip />} />
                    <Legend />
                    <Line type="monotone" dataKey="loss" stroke="hsl(var(--destructive))" strokeWidth={2} dot={{ r: 3 }} name="Loss" connectNulls />
                  </ComposedChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Recommendation + best config */}
      {hasInsights && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2"><Target className="h-4 w-4 text-green-500" /> Best Configuration Found</CardTitle>
              <CardDescription>Hyperparameters that produced your best results</CardDescription>
            </CardHeader>
            <CardContent>
              {insights.insights?.best_config ? (
                <div className="space-y-3">
                  {Object.entries(insights.insights.best_config).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground capitalize">{key.replace(/_/g, ' ')}</span>
                      <span className="font-mono text-sm font-medium">{String(value)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No best config available yet.</p>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2"><Lightbulb className="h-4 w-4 text-amber-500" /> Recommendation</CardTitle>
              <CardDescription>What to try next</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm">{insights.insights?.recommendation || 'Keep training to generate recommendations.'}</p>
              {insights.insights?.best_config && (
                <Button size="sm" className="mt-3" onClick={() => { void insights.applyRecommendedConfig() }} disabled={insights.useConfigStatus === 'applied'}>
                  {insights.useConfigStatus === 'applied' ? 'Applied!' : 'Use recommended config'}
                </Button>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Distributions + sparkline */}
      {analytics.analytics && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card><CardContent className="p-6"><h3 className="text-lg font-semibold mb-4">Quality Trend</h3><QualitySparkline data={analytics.analytics.quality_trend} /></CardContent></Card>
          <Card><CardContent className="p-6"><h3 className="text-lg font-semibold mb-4">By Method</h3><DistributionBars data={methodData} /></CardContent></Card>
          <Card><CardContent className="p-6"><h3 className="text-lg font-semibold mb-4">By Model</h3><DistributionBars data={modelData} /></CardContent></Card>
          <Card><CardContent className="p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2"><Tag className="h-4 w-4" /> Tags</h3>
            <div className="flex flex-wrap gap-2">
              {Object.entries(analytics.analytics.tag_cloud).sort((a, b) => b[1] - a[1]).map(([tag, count]) => (
                <Badge key={tag} className="bg-violet-100 text-violet-800" title={`${count} run${count !== 1 ? 's' : ''}`}>{tag}</Badge>
              ))}
              {Object.keys(analytics.analytics.tag_cloud).length === 0 && <p className="text-sm text-muted-foreground">No tags yet</p>}
            </div>
          </CardContent></Card>
        </div>
      )}

      {/* Model breakdown table */}
      {trends.models.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Model Breakdown</CardTitle>
            <CardDescription>Performance by model</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/50">
                    <th className="text-left py-2 text-xs text-muted-foreground font-medium">Model</th>
                    <th className="text-right py-2 text-xs text-muted-foreground font-medium">Runs</th>
                    <th className="text-right py-2 text-xs text-muted-foreground font-medium">Avg Quality</th>
                    <th className="text-right py-2 text-xs text-muted-foreground font-medium">Best</th>
                  </tr>
                </thead>
                <tbody>
                  {trends.models.map(m => (
                    <tr key={m.model} className="border-b border-border/30">
                      <td className="py-2 font-mono text-xs">{m.model}</td>
                      <td className="py-2 text-right">{m.total_runs}</td>
                      <td className="py-2 text-right font-mono">{m.avg_quality.toFixed(3)}</td>
                      <td className="py-2 text-right font-mono">{m.best_quality.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Best run */}
      {analytics.analytics?.best_run && (
        <Card>
          <CardContent className="p-6">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2"><Award className="h-4 w-4 text-amber-500" /> Best Run</h3>
            <div className="flex items-center gap-4 text-sm">
              <span className="font-mono text-muted-foreground">{analytics.analytics.best_run.run_id}</span>
              <QualityBadge score={analytics.analytics.best_run.quality_score} />
              <span>{analytics.analytics.best_run.model}</span>
              <span className="text-muted-foreground">{analytics.analytics.best_run.method}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Run history table */}
      {trends.runs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Run History</CardTitle>
            <CardDescription>All recorded training runs (newest first)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/50">
                    <th className="text-left py-2 text-xs text-muted-foreground font-medium">Run</th>
                    <th className="text-left py-2 text-xs text-muted-foreground font-medium">Date</th>
                    <th className="text-left py-2 text-xs text-muted-foreground font-medium">Model</th>
                    <th className="text-left py-2 text-xs text-muted-foreground font-medium">Dataset</th>
                    <th className="text-left py-2 text-xs text-muted-foreground font-medium">Method</th>
                    <th className="text-right py-2 text-xs text-muted-foreground font-medium">Quality</th>
                    <th className="text-right py-2 text-xs text-muted-foreground font-medium">Loss</th>
                    <th className="text-center py-2 text-xs text-muted-foreground font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {[...trends.runs].reverse().map(r => (
                    <tr key={r.run_id} className="border-b border-border/30 hover:bg-muted/30">
                      <td className="py-2 font-mono text-xs">{r.run_id.slice(-12)}</td>
                      <td className="py-2 text-xs">{formatRunDate(r.timestamp)}</td>
                      <td className="py-2 font-mono text-xs">{r.model || '—'}</td>
                      <td className="py-2 text-xs">{r.dataset || '—'}</td>
                      <td className="py-2 text-xs">{r.method || '—'}</td>
                      <td className="py-2 text-right font-mono">{r.quality_score.toFixed(3)}</td>
                      <td className="py-2 text-right font-mono">{r.final_loss > 0 ? r.final_loss.toFixed(4) : '—'}</td>
                      <td className="py-2 text-center"><RunStatusBadge converged={r.converged} earlyStopped={r.early_stopped} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Per-method loss bar chart (from analytics distributions) */}
      {methodData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2"><BarChart3 className="h-4 w-4" /> Runs by Method</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={methodData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                <Tooltip content={<TrainingChartTooltip />} />
                <Bar dataKey="value" fill="hsl(var(--primary))" name="Runs" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {analytics.analytics && (
        <div className="flex justify-end">
          <button onClick={() => analytics.analytics && downloadJson(analytics.analytics, `training-analytics-${Date.now()}.json`)} className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
            <IconDownload className="h-3 w-3" /> Export JSON
          </button>
        </div>
      )}
    </div>
  )
}
