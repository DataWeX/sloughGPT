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
} from 'recharts'
import { apiGet } from '@/lib/http-client'
import { settingsController, type AdaptiveInsights } from '@/lib/settings-controller'
import { TrendingUp, TrendingDown, Minus, Brain, Lightbulb, Target, Sparkles, RefreshCw } from 'lucide-react'

interface TrendRun {
  run_id: string
  timestamp: number
  model: string
  dataset: string
  method: string
  quality_score: number
  final_loss: number
  learning_rate: number
  batch_size: number
  epochs: number
}

interface TrendData {
  runs: TrendRun[]
  summary: {
    total_runs: number
    avg_quality: number
    best_quality: number
    avg_loss: number
    trend: string
  }
}

function formatDate(ts: number): string {
  if (!ts) return '—'
  const d = new Date(ts * 1000)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function TrendIcon({ trend }: { trend: string }) {
  if (trend === 'improving') return <TrendingUp className="h-5 w-5 text-green-500" />
  if (trend === 'declining') return <TrendingDown className="h-5 w-5 text-red-500" />
  return <Minus className="h-5 w-5 text-muted-foreground" />
}

function InsightTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }>; label?: string }) {
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

export default function TrainingInsightsPage() {
  const [insights, setInsights] = useState<AdaptiveInsights | null>(null)
  const [trendData, setTrendData] = useState<TrendData | null>(null)
  const [loading, setLoading] = useState(true)
  const [useConfigStatus, setUseConfigStatus] = useState<'idle' | 'applied'>('idle')
  const [exportStatus, setExportStatus] = useState<'idle' | 'exporting'>('idle')

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const [insightsResp, trendResp] = await Promise.all([
        settingsController.getAdaptiveInsights().catch(() => null),
        apiGet<TrendData>('/training/trends').catch(() => null),
      ])
      setInsights(insightsResp)
      setTrendData(trendResp)
    } catch (err) {
      console.error('Failed to load insights:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadData() }, [loadData])

  const handleUseRecommended = async () => {
    if (!insights?.best_config) return
    try {
      await settingsController.updateTraining({
        preferred_model: insights.best_config.model as unknown as string || 'gpt2',
        preferred_method: 'finetune',
        max_checkpoints: 10,
      })
      setUseConfigStatus('applied')
      setTimeout(() => setUseConfigStatus('idle'), 3000)
    } catch {
      // silently fail
    }
  }

  const chartData = trendData?.runs.map((r, i) => ({
    name: `#${i + 1}`,
    date: formatDate(r.timestamp),
    quality: r.quality_score,
    loss: r.final_loss > 0 ? r.final_loss : null,
    model: r.model,
  })) ?? []

  const hasData = insights && insights.total_runs !== undefined && insights.total_runs > 0
  const hasMessage = insights?.message

  return (
    <PageContainer title="Adaptive Insights">
      <AppRouteHeader left={<AppRouteHeaderLead title="Adaptive Insights" />} />

      {loading ? (
        <div className="space-y-4">
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : hasMessage && !hasData ? (
        <Card>
          <CardContent className="py-16 text-center">
            <Brain className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <p className="text-lg font-medium mb-2">No Training History Yet</p>
            <p className="text-sm text-muted-foreground max-w-md mx-auto">
              {insights.message}
            </p>
            <p className="text-xs text-muted-foreground mt-4">
              Complete a training run to start building adaptive intelligence.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          {/* Summary Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-2 mb-1">
                  <Target className="h-4 w-4 text-muted-foreground" />
                  <p className="text-xs text-muted-foreground">Total Runs</p>
                </div>
                <p className="text-3xl font-bold">{insights?.total_runs ?? 0}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-2 mb-1">
                  <Sparkles className="h-4 w-4 text-muted-foreground" />
                  <p className="text-xs text-muted-foreground">Avg Quality</p>
                </div>
                <p className="text-3xl font-bold">{(insights?.avg_quality ?? 0).toFixed(3)}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingUp className="h-4 w-4 text-green-500" />
                  <p className="text-xs text-muted-foreground">Best Quality</p>
                </div>
                <p className="text-3xl font-bold text-green-500">{(insights?.best_quality ?? 0).toFixed(3)}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-2 mb-1">
                  <TrendIcon trend={insights?.trend ?? 'stable'} />
                  <p className="text-xs text-muted-foreground">Trend</p>
                </div>
                <p className="text-3xl font-bold capitalize">{insights?.trend ?? '—'}</p>
              </CardContent>
            </Card>
          </div>

          {/* Learning Curve */}
          {chartData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Brain className="h-4 w-4" /> Learning Curve
                </CardTitle>
                <CardDescription>Quality score over consecutive training runs</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={280}>
                  <ComposedChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                    <Tooltip content={<InsightTooltip />} />
                    <Legend />
                    <Area type="monotone" dataKey="quality" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.1} name="Quality" strokeWidth={2} dot={{ r: 3 }} />
                  </ComposedChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Best Config + Recommendation */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Best Config Found */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Target className="h-4 w-4 text-green-500" /> Best Configuration Found
                </CardTitle>
                <CardDescription>Hyperparameters that produced your best results</CardDescription>
              </CardHeader>
              <CardContent>
                {insights?.best_config ? (
                  <div className="space-y-3">
                    {Object.entries(insights.best_config).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground capitalize">{key.replace(/_/g, ' ')}</span>
                        <span className="font-mono text-sm font-medium">{String(value)}</span>
                      </div>
                    ))}
                    <div className="pt-2 border-t border-border/50">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Achieved Quality</span>
                        <Badge variant="default" className="font-mono">{(insights?.best_quality ?? 0).toFixed(3)}</Badge>
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No best config available yet.</p>
                )}
              </CardContent>
            </Card>

            {/* Next Recommendation */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Lightbulb className="h-4 w-4 text-yellow-500" /> Next Recommended Config
                </CardTitle>
                <CardDescription>What the adaptive engine suggests for your next run</CardDescription>
              </CardHeader>
              <CardContent>
                {insights?.recommendation ? (
                  <div className="space-y-4">
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {insights.recommendation}
                    </p>
                    <Button
                      variant={useConfigStatus === 'applied' ? 'default' : 'outline'}
                      size="sm"
                      onClick={handleUseRecommended}
                      disabled={useConfigStatus === 'applied'}
                    >
                      {useConfigStatus === 'applied' ? (
                        <>Applied to Training Settings</>
                      ) : (
                        <>Use Recommended Config</>
                      )}
                    </Button>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">Run more training to get recommendations.</p>
                )}
              </CardContent>
            </Card>
          </div>

          {/* What the Engine Learned */}
          {hasData && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Sparkles className="h-4 w-4" /> What the Engine Learned
                </CardTitle>
                <CardDescription>Patterns discovered from your training history</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">Data Efficiency</p>
                    <p className="text-sm">
                      {(insights?.total_runs ?? 0) >= 5
                        ? `With ${insights?.total_runs} runs, the engine has enough data to make confident recommendations.`
                        : `Only ${insights?.total_runs} run(s) recorded. The engine needs ~5 runs for confident predictions.`}
                    </p>
                  </div>
                  <div className="space-y-2">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">Quality Trajectory</p>
                    <p className="text-sm">
                      {insights?.trend === 'improving'
                        ? 'Your model quality is improving over time. Keep training!'
                        : insights?.trend === 'declining'
                        ? 'Quality has declined recently. Consider reviewing hyperparameters.'
                        : 'Insufficient data to determine quality trajectory.'}
                    </p>
                  </div>
                  <div className="space-y-2">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">Avg Loss</p>
                    <p className="text-sm font-mono">
                      {(insights?.avg_loss ?? 0) > 0 ? insights?.avg_loss?.toFixed(4) : '—'}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {(insights?.avg_loss ?? 0) > 0 && insights?.avg_loss! < 3
                        ? 'Good — loss is in a healthy range'
                        : (insights?.avg_loss ?? 0) >= 3
                        ? 'High — consider more epochs or lower learning rate'
                        : 'No loss data available'}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Refresh */}
          <div className="flex justify-end">
            <Button variant="outline" size="sm" onClick={loadData}>
              <RefreshCw className="h-4 w-4 mr-1" /> Refresh
            </Button>
          </div>
        </div>
      )}
    </PageContainer>
  )
}
