'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Badge,
  Button,
  Skeleton,
  Input,
} from '@sloughgpt/strui'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  Area,
  ComposedChart,
  Bar,
  BarChart,
} from 'recharts'
import { apiGet } from '@/lib/http-client'
import { TrendingUp, TrendingDown, Minus, RefreshCw } from 'lucide-react'

interface TrendRun {
  run_id: string
  timestamp: number
  dataset: string
  dataset_size: number
  model: string
  method: string
  epochs: number
  batch_size: number
  learning_rate: number
  final_loss: number
  best_loss: number
  perplexity: number
  converged: boolean
  early_stopped: boolean
  quality_score: number
  training_time_s: number
}

interface TrendModel {
  model: string
  total_runs: number
  avg_quality: number
  best_quality: number
  avg_loss: number
  latest_quality: number
  improving: boolean
}

interface TrendSummary {
  total_runs: number
  avg_quality: number
  best_quality: number
  avg_loss: number
  trend: 'improving' | 'declining' | 'stable'
}

interface TrendData {
  runs: TrendRun[]
  models: TrendModel[]
  summary: TrendSummary
}

function formatDate(ts: number): string {
  if (!ts) return '—'
  const d = new Date(ts * 1000)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function formatTime(ts: number): string {
  if (!ts) return '—'
  const d = new Date(ts * 1000)
  return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
}

function TrendIcon({ trend }: { trend: string }) {
  if (trend === 'improving') return <TrendingUp className="h-4 w-4 text-green-500" />
  if (trend === 'declining') return <TrendingDown className="h-4 w-4 text-red-500" />
  return <Minus className="h-4 w-4 text-muted-foreground" />
}

function TrendTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }>; label?: string }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-card border border-border rounded-lg px-3 py-2 shadow-lg text-xs">
      <p className="font-medium mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </p>
      ))}
    </div>
  )
}

export default function TrainingTrendsPage() {
  const [data, setData] = useState<TrendData | null>(null)
  const [loading, setLoading] = useState(true)
  const [modelFilter, setModelFilter] = useState('')
  const [datasetFilter, setDatasetFilter] = useState('')

  const loadTrends = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (modelFilter) params.set('model', modelFilter)
      if (datasetFilter) params.set('dataset', datasetFilter)
      const qs = params.toString()
      const resp = await apiGet<TrendData>(`/training/trends${qs ? `?${qs}` : ''}`)
      setData(resp)
    } catch (err) {
      console.error('Failed to load trends:', err)
    } finally {
      setLoading(false)
    }
  }, [modelFilter, datasetFilter])

  useEffect(() => { loadTrends() }, [loadTrends])

  const chartData = data?.runs.map(r => ({
    name: `${r.model || 'unknown'}#${r.run_id.slice(-6)}`,
    date: formatDate(r.timestamp),
    quality: r.quality_score,
    loss: r.final_loss > 0 ? r.final_loss : null,
    perplexity: r.perplexity > 0 ? r.perplexity : null,
    dataset: r.dataset,
    model: r.model,
    method: r.method,
  })) ?? []

  return (
    <PageContainer title="Training Trends">
      <AppRouteHeader left={<AppRouteHeaderLead title="Training Trends" />} />

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground">Total Runs</p>
            <p className="text-2xl font-bold mt-1">{loading ? '—' : data?.summary.total_runs ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground">Avg Quality</p>
            <p className="text-2xl font-bold mt-1">{loading ? '—' : (data?.summary.avg_quality ?? 0).toFixed(3)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground">Best Quality</p>
            <p className="text-2xl font-bold mt-1 text-green-500">{loading ? '—' : (data?.summary.best_quality ?? 0).toFixed(3)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <p className="text-xs text-muted-foreground">Trend</p>
              <TrendIcon trend={data?.summary.trend ?? 'stable'} />
            </div>
            <p className="text-2xl font-bold mt-1 capitalize">{loading ? '—' : data?.summary.trend ?? 'stable'}</p>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 mb-6">
        <Input
          value={modelFilter}
          onChange={e => setModelFilter(e.target.value)}
          placeholder="Filter by model..."
          className="w-48"
        />
        <Input
          value={datasetFilter}
          onChange={e => setDatasetFilter(e.target.value)}
          placeholder="Filter by dataset..."
          className="w-48"
        />
        <Button variant="outline" size="sm" onClick={loadTrends}>
          <RefreshCw className="h-4 w-4 mr-1" /> Refresh
        </Button>
      </div>

      {loading ? (
        <div className="space-y-4">
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : !data || data.runs.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-muted-foreground">No training runs recorded yet.</p>
            <p className="text-xs text-muted-foreground mt-1">Complete a training run to see trends here.</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          {/* Quality Over Time */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Quality Over Time</CardTitle>
              <CardDescription>Training quality score per run (0-1, higher is better)</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip content={<TrendTooltip />} />
                  <Legend />
                  <Area type="monotone" dataKey="quality" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.1} name="Quality" />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Loss Over Time */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Loss Over Time</CardTitle>
              <CardDescription>Final training loss per run (lower is better)</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip content={<TrendTooltip />} />
                  <Legend />
                  <Line type="monotone" dataKey="loss" stroke="hsl(var(--destructive))" strokeWidth={2} dot={{ r: 3 }} name="Loss" connectNulls />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Per-Model Breakdown */}
          {data.models.length > 0 && (
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
                        <th className="text-right py-2 text-xs text-muted-foreground font-medium">Best Quality</th>
                        <th className="text-right py-2 text-xs text-muted-foreground font-medium">Avg Loss</th>
                        <th className="text-right py-2 text-xs text-muted-foreground font-medium">Latest</th>
                        <th className="text-center py-2 text-xs text-muted-foreground font-medium">Trend</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.models.map(m => (
                        <tr key={m.model} className="border-b border-border/30 hover:bg-muted/30">
                          <td className="py-2 font-mono text-xs">{m.model || 'unknown'}</td>
                          <td className="py-2 text-right">{m.total_runs}</td>
                          <td className="py-2 text-right font-mono">{m.avg_quality.toFixed(3)}</td>
                          <td className="py-2 text-right font-mono text-green-500">{m.best_quality.toFixed(3)}</td>
                          <td className="py-2 text-right font-mono">{m.avg_loss > 0 ? m.avg_loss.toFixed(4) : '—'}</td>
                          <td className="py-2 text-right font-mono">{m.latest_quality.toFixed(3)}</td>
                          <td className="py-2 text-center">
                            {m.improving ? (
                              <Badge variant="default" className="text-[10px]">↑ improving</Badge>
                            ) : (
                              <Badge variant="secondary" className="text-[10px]">— stable</Badge>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Run History */}
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
                    {[...data.runs].reverse().map(r => (
                      <tr key={r.run_id} className="border-b border-border/30 hover:bg-muted/30">
                        <td className="py-2 font-mono text-xs">{r.run_id.slice(-12)}</td>
                        <td className="py-2 text-xs">{formatDate(r.timestamp)} {formatTime(r.timestamp)}</td>
                        <td className="py-2 font-mono text-xs">{r.model || '—'}</td>
                        <td className="py-2 text-xs">{r.dataset || '—'}</td>
                        <td className="py-2 text-xs">{r.method || '—'}</td>
                        <td className="py-2 text-right font-mono">{r.quality_score.toFixed(3)}</td>
                        <td className="py-2 text-right font-mono">{r.final_loss > 0 ? r.final_loss.toFixed(4) : '—'}</td>
                        <td className="py-2 text-center">
                          {r.converged ? (
                            <Badge variant="default" className="text-[10px]">converged</Badge>
                          ) : r.early_stopped ? (
                            <Badge variant="destructive" className="text-[10px]">early stopped</Badge>
                          ) : (
                            <Badge variant="secondary" className="text-[10px]">completed</Badge>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </PageContainer>
  )
}
