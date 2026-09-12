'use client'

import { useState, useEffect, useCallback, useMemo } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle, Skeleton,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'

interface Episode {
  input: string
  response: string
  narrative: string
  qualia: Record<string, number>
  growth_delta: number
  rating: number
  timestamp: string
}

interface StatusData {
  response_quality: number
  last_reflection: string
}

interface HealthData {
  health_score: number
  episodes: number
  avg_growth: number
  positive_ratio: number
}

interface EvalData {
  overall_score: number
  response_quality: number
  narrative_quality: number
  belief_consistency: number
  qualia_coherence: number
  diagnostics: string[]
}

const QUALIA_KEYS = ['valence', 'arousal', 'novelty', 'coherence', 'salience', 'certainty', 'complexity']
const QUALIA_COLORS: Record<string, string> = {
  valence: '#8b5cf6', arousal: '#ef4444', novelty: '#f59e0b',
  coherence: '#22c55e', salience: '#3b82f6', certainty: '#06b6d4', complexity: '#d946ef',
}
const RATING_COLORS = ['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e']

const SVG_W = 800
const SVG_H = 300
const M = { top: 20, right: 20, bottom: 40, left: 60 }
const PW = SVG_W - M.left - M.right
const PH = SVG_H - M.top - M.bottom

function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)) }

function movingAverage(data: number[], window: number): (number | null)[] {
  const result: (number | null)[] = []
  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - Math.floor(window / 2))
    const end = Math.min(data.length, i + Math.ceil(window / 2))
    const slice = data.slice(start, end)
    result.push(slice.reduce((a, b) => a + b, 0) / slice.length)
  }
  return result
}

function buildLinePath(data: number[], yMin: number, yMax: number): string {
  if (data.length === 0) return ''
  const xStep = data.length > 1 ? PW / (data.length - 1) : PW / 2
  return data.map((v, i) => {
    const x = M.left + (data.length > 1 ? i * xStep : PW / 2)
    const y = M.top + PH - ((v - yMin) / (yMax - yMin)) * PH
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

function buildAreaPath(data: number[], yMin: number, yMax: number, baseline: number): string {
  if (data.length === 0) return ''
  const xStep = data.length > 1 ? PW / (data.length - 1) : PW / 2
  const baseY = M.top + PH - ((baseline - yMin) / (yMax - yMin)) * PH
  const points = data.map((v, i) => {
    const x = M.left + (data.length > 1 ? i * xStep : PW / 2)
    const y = M.top + PH - ((v - yMin) / (yMax - yMin)) * PH
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })
  const firstX = M.left + (data.length > 1 ? 0 : PW / 2)
  const lastX = M.left + (data.length > 1 ? (data.length - 1) * xStep : PW / 2)
  return `M${firstX.toFixed(1)},${baseY.toFixed(1)} L${points.join(' L')} L${lastX.toFixed(1)},${baseY.toFixed(1)} Z`
}

function GridLines({ yMin, yMax, steps }: { yMin: number; yMax: number; steps: number }) {
  const lines = []
  for (let i = 0; i <= steps; i++) {
    const y = M.top + (PH / steps) * i
    const val = yMax - (yMax - yMin) * (i / steps)
    lines.push(
      <g key={i}>
        <line x1={M.left} y1={y} x2={M.left + PW} y2={y} stroke="hsl(var(--border))" strokeWidth="0.5" />
        <text x={M.left - 8} y={y + 3} textAnchor="end" fontSize="10" className="fill-muted-foreground">
          {val.toFixed(2)}
        </text>
      </g>
    )
  }
  return <>{lines}</>
}

function XAxisLabels({ labels }: { labels: string[] }) {
  if (labels.length === 0) return null
  const step = Math.max(1, Math.floor(labels.length / 8))
  return <>{labels.map((l, i) => i % step === 0 ? (
    <text key={i} x={M.left + (labels.length > 1 ? (i / (labels.length - 1)) * PW : PW / 2)} y={SVG_H - 8} textAnchor="middle" fontSize="10" className="fill-muted-foreground">{l}</text>
  ) : null)}</>
}

function TooltipBox({ x, y, lines }: { x: number; y: number; lines: string[] }) {
  const boxW = 160
  const boxH = lines.length * 16 + 10
  const bx = clamp(x - boxW / 2, 0, SVG_W - boxW)
  const by = clamp(y - boxH - 8, 0, SVG_H - boxH)
  return (
    <g>
      <rect x={bx} y={by} width={boxW} height={boxH} rx="4" fill="hsl(var(--card))" stroke="hsl(var(--border))" strokeWidth="1" />
      {lines.map((line, i) => (
        <text key={i} x={bx + 8} y={by + 14 + i * 16} fontSize="10" className="fill-foreground">{line}</text>
      ))}
    </g>
  )
}

function BarChart({ data, labels, colors, yLabel, tooltip, onTooltip }: {
  data: number[]
  labels: string[]
  colors: string[]
  yLabel: string
  tooltip?: { x: number; y: number; lines: string[] } | null
  onTooltip?: (t: { x: number; y: number; lines: string[] } | null) => void
}) {
  const yMax = Math.max(1, ...data)
  return (
    <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full h-auto">
      <GridLines yMin={0} yMax={yMax} steps={4} />
      <text x={M.left - 40} y={M.top + PH / 2} textAnchor="middle" fontSize="10" className="fill-muted-foreground" transform={`rotate(-90, ${M.left - 40}, ${M.top + PH / 2})`}>{yLabel}</text>
      <text x={M.left + PW / 2} y={SVG_H - 2} textAnchor="middle" fontSize="10" className="fill-muted-foreground">Day</text>
      {data.map((count, i) => {
        const barW = Math.max(8, Math.min(40, PW / data.length - 4))
        const x = M.left + (PW / data.length) * i + (PW / data.length - barW) / 2
        const barH = (count / yMax) * PH
        const y = M.top + PH - barH
        return (
          <g key={i}>
            <rect x={x} y={y} width={barW} height={barH} rx="2" fill={colors[i % colors.length]} fillOpacity="0.8"
              onMouseEnter={() => onTooltip?.({ x: x + barW / 2, y, lines: [labels[i], `Count: ${count}`] })}
              onMouseLeave={() => onTooltip?.(null)}
            />
            <text x={x + barW / 2} y={SVG_H - 8} textAnchor="middle" fontSize="9" className="fill-muted-foreground">{labels[i]}</text>
          </g>
        )
      })}
      {tooltip && <TooltipBox x={tooltip.x} y={tooltip.y} lines={tooltip.lines} />}
    </svg>
  )
}

export default function ConsciousnessStatisticsPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()
  const [loading, setLoading] = useState(true)
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [statusData, setStatusData] = useState<StatusData | null>(null)
  const [healthData, setHealthData] = useState<HealthData | null>(null)
  const [evalData, setEvalData] = useState<EvalData | null>(null)

  const [tooltipBar, setTooltipBar] = useState<{ x: number; y: number; lines: string[] } | null>(null)
  const [tooltipLine1, setTooltipLine1] = useState<{ x: number; y: number; lines: string[] } | null>(null)
  const [tooltipLine2, setTooltipLine2] = useState<{ x: number; y: number; lines: string[] } | null>(null)

  const fetchAll = useCallback(async () => {
    try {
      const [epRes, statusRes, healthRes, evalRes] = await Promise.allSettled([
        consciousnessController.getEpisodeHistory(100),
        consciousnessController.getStatus(),
        consciousnessController.healthCheck(),
        consciousnessController.evaluate(),
      ])

      if (epRes.status === 'fulfilled') {
        setEpisodes((epRes.value.episodes ?? []) as unknown as Episode[])
      }
      if (statusRes.status === 'fulfilled') {
        setStatusData(statusRes.value as unknown as StatusData)
      }
      if (healthRes.status === 'fulfilled') {
        setHealthData(healthRes.value as unknown as HealthData)
      }
      if (evalRes.status === 'fulfilled') {
        setEvalData(evalRes.value as unknown as EvalData)
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { fetchAll() }, [fetchAll])

  const healthScore = healthData?.health_score ?? evalData?.overall_score ?? 0

  const trend = useMemo(() => {
    if (episodes.length < 2) return 'stable' as const
    const growths = episodes.map(e => e.growth_delta)
    const half = Math.floor(growths.length / 2)
    const firstHalf = growths.slice(0, half).reduce((a, b) => a + b, 0) / (half || 1)
    const secondHalf = growths.slice(half).reduce((a, b) => a + b, 0) / ((growths.length - half) || 1)
    if (secondHalf > firstHalf + 0.005) return 'improving' as const
    if (secondHalf < firstHalf - 0.005) return 'declining' as const
    return 'stable' as const
  }, [episodes])

  const performanceMetrics = useMemo(() => {
    if (episodes.length === 0) return null
    const ratings = episodes.map(e => e.rating).filter(r => r > 0)
    const ratingCounts = [0, 0, 0, 0, 0]
    ratings.forEach(r => { if (r >= 1 && r <= 5) ratingCounts[r - 1]++ })

    const qualiaDiversity = new Set<string>()
    episodes.forEach(e => {
      if (e.qualia) {
        Object.entries(e.qualia).forEach(([k, v]) => {
          if (typeof v === 'number' && v > 0.1) qualiaDiversity.add(k)
        })
      }
    })

    const timestamps = episodes.map(e => new Date(e.timestamp).getTime()).sort((a, b) => a - b)
    let episodesPerDay = 0
    if (timestamps.length > 1) {
      const spanMs = timestamps[timestamps.length - 1] - timestamps[0]
      const spanDays = Math.max(1, spanMs / (1000 * 60 * 60 * 24))
      episodesPerDay = episodes.length / spanDays
    } else {
      episodesPerDay = episodes.length
    }

    const avgGrowth = episodes.reduce((a, e) => a + e.growth_delta, 0) / episodes.length
    const avgRating = ratings.length > 0 ? ratings.reduce((a, b) => a + b, 0) / ratings.length : 0
    const positiveRatio = episodes.filter(e => e.growth_delta > 0).length / episodes.length

    return {
      episodesPerDay,
      avgGrowth,
      avgRating,
      positiveRatio,
      ratingCounts,
      qualiaDiversity: qualiaDiversity.size,
      totalDimensions: 7,
    }
  }, [episodes])

  const episodesByDay = useMemo(() => {
    const map = new Map<string, number>()
    episodes.forEach(e => {
      const day = e.timestamp.slice(0, 10)
      map.set(day, (map.get(day) ?? 0) + 1)
    })
    const sorted = Array.from(map.entries()).sort((a, b) => a[0].localeCompare(b[0]))
    return {
      counts: sorted.map(([, c]) => c),
      labels: sorted.map(([d]) => d.slice(5)),
    }
  }, [episodes])

  const avgRatingByDay = useMemo(() => {
    const map = new Map<string, { sum: number; count: number }>()
    episodes.forEach(e => {
      if (e.rating <= 0) return
      const day = e.timestamp.slice(0, 10)
      const existing = map.get(day) ?? { sum: 0, count: 0 }
      existing.sum += e.rating
      existing.count++
      map.set(day, existing)
    })
    const sorted = Array.from(map.entries()).sort((a, b) => a[0].localeCompare(b[0]))
    return {
      values: sorted.map(([, v]) => v.sum / v.count),
      labels: sorted.map(([d]) => d.slice(5)),
    }
  }, [episodes])

  const growthTrend = useMemo(() => {
    const values = episodes.map(e => e.growth_delta)
    const avg = movingAverage(values, 5)
    const labels = episodes.map((_, i) => `${i + 1}`)
    return { values, avg, labels }
  }, [episodes])

  const topQualia = useMemo(() => {
    const counts: Record<string, number> = {}
    episodes.forEach(e => {
      if (e.qualia) {
        Object.entries(e.qualia).forEach(([k, v]) => {
          if (typeof v === 'number' && v > 0.3) counts[k] = (counts[k] ?? 0) + 1
        })
      }
    })
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 5)
  }, [episodes])

  const bestWorst = useMemo(() => {
    if (episodes.length === 0) return { best: null, worst: null }
    let best = episodes[0]
    let worst = episodes[0]
    episodes.forEach(e => {
      if (e.rating > best.rating || (e.rating === best.rating && e.growth_delta > best.growth_delta)) best = e
      if (e.rating < worst.rating || (e.rating === worst.rating && e.growth_delta < worst.growth_delta)) worst = e
    })
    return { best, worst }
  }, [episodes])

  const beliefTrends = useMemo(() => {
    if (episodes.length < 2) return []
    const mid = Math.floor(episodes.length / 2)
    const firstHalf = episodes.slice(0, mid)
    const secondHalf = episodes.slice(mid)
    const qualiaKeys = ['valence', 'arousal', 'novelty', 'coherence', 'salience', 'certainty', 'complexity']
    return qualiaKeys.map(key => {
      const firstAvg = firstHalf.length > 0 ? firstHalf.reduce((a, e) => a + (e.qualia?.[key] ?? 0), 0) / firstHalf.length : 0
      const secondAvg = secondHalf.length > 0 ? secondHalf.reduce((a, e) => a + (e.qualia?.[key] ?? 0), 0) / secondHalf.length : 0
      const delta = secondAvg - firstAvg
      return { key, delta, growing: delta > 0.01, shrinking: delta < -0.01 }
    }).filter(b => b.growing || b.shrinking)
  }, [episodes])

  const handleExport = useCallback(() => {
    const exportData = {
      exported: new Date().toISOString(),
      health_score: healthScore,
      trend,
      episodes_total: episodes.length,
      performance: performanceMetrics,
      episodes_by_day: episodesByDay,
      avg_rating_by_day: avgRatingByDay,
      top_qualia: topQualia,
      belief_trends: beliefTrends,
      health_data: healthData,
      eval_data: evalData,
      status_data: statusData,
      episodes: episodes.map(e => ({
        timestamp: e.timestamp,
        rating: e.rating,
        growth_delta: e.growth_delta,
        qualia: e.qualia,
      })),
    }
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `consciousness-statistics-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
    addToast('Statistics exported', 'success')
  }, [healthScore, trend, episodes, performanceMetrics, episodesByDay, avgRatingByDay, topQualia, beliefTrends, healthData, evalData, statusData, addToast])

  const healthColor = healthScore > 70 ? 'text-green-500' : healthScore > 40 ? 'text-yellow-500' : 'text-red-500'
  const healthBg = healthScore > 70 ? 'bg-green-500/10' : healthScore > 40 ? 'bg-yellow-500/10' : 'bg-red-500/10'

  if (loading) {
    return (
      <PageContainer title={t('consciousness_statistics.page_title')}>
        <div className="space-y-6 p-6">
          <Skeleton className="h-48" />
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
          </div>
          <Skeleton className="h-72" />
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Skeleton className="h-72" />
            <Skeleton className="h-72" />
          </div>
        </div>
      </PageContainer>
    )
  }

  return (
    <PageContainer title={t('consciousness_statistics.page_title')}>
      <div className="space-y-6 p-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">{t('consciousness_statistics.health_score')}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center">
              <div className={`relative flex items-center justify-center w-32 h-32 rounded-full ${healthBg}`}>
                <span className={`text-4xl font-bold ${healthColor}`}>{healthScore}</span>
                <span className="absolute bottom-1 text-xs text-muted-foreground">/ 100</span>
              </div>
            </div>
            <div className="flex items-center justify-center mt-4 gap-6 text-sm text-muted-foreground">
              <span>{t('consciousness_statistics.trend')}: <span className={`font-medium ${
                trend === 'improving' ? 'text-green-500' : trend === 'declining' ? 'text-red-500' : 'text-muted-foreground'
              }`}>
                {t(`consciousness_statistics.trend_${trend}`)}
              </span></span>
              <span>{t('consciousness_statistics.episodes')}: {healthData?.episodes ?? episodes.length}</span>
              <span>{t('consciousness_statistics.avg_growth')}: {((healthData?.avg_growth ?? 0) * 100).toFixed(1)}%</span>
              <span>{t('consciousness_statistics.positive_ratio')}: {((healthData?.positive_ratio ?? 0) * 100).toFixed(0)}%</span>
            </div>
          </CardContent>
        </Card>

        {performanceMetrics && (
          <div className="grid grid-cols-2 gap-4 md:grid-cols-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-muted-foreground">{t('consciousness_statistics.episodes_per_day')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{performanceMetrics.episodesPerDay.toFixed(1)}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-muted-foreground">{t('consciousness_statistics.avg_growth_ep')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold ${performanceMetrics.avgGrowth >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {performanceMetrics.avgGrowth >= 0 ? '+' : ''}{(performanceMetrics.avgGrowth * 100).toFixed(1)}%
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-muted-foreground">{t('consciousness_statistics.avg_rating')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{performanceMetrics.avgRating.toFixed(1)} / 5</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-muted-foreground">{t('consciousness_statistics.qualia_diversity')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{performanceMetrics.qualiaDiversity}/{performanceMetrics.totalDimensions}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-muted-foreground">{t('consciousness_statistics.positive_ratio')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{(performanceMetrics.positiveRatio * 100).toFixed(0)}%</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-muted-foreground">{t('consciousness_statistics.belief_stability')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{evalData?.belief_consistency != null ? `${(evalData.belief_consistency * 100).toFixed(0)}%` : '—'}</div>
              </CardContent>
            </Card>
          </div>
        )}

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_statistics.rating_dist_title')}</CardTitle>
            <CardDescription>{t('consciousness_statistics.rating_dist_desc')}</CardDescription>
          </CardHeader>
          <CardContent>
            {performanceMetrics ? (
              <div className="flex items-end justify-center gap-4 h-32">
                {performanceMetrics.ratingCounts.map((count, i) => {
                  const maxCount = Math.max(1, ...performanceMetrics.ratingCounts)
                  const h = (count / maxCount) * 100
                  return (
                    <div key={i} className="flex flex-col items-center gap-1">
                      <span className="text-xs text-muted-foreground">{count}</span>
                      <div className="w-12 rounded-t" style={{ height: `${h}%`, backgroundColor: RATING_COLORS[i], opacity: 0.8 }} />
                      <div className="flex gap-0.5">
                        {Array.from({ length: i + 1 }).map((_, s) => (
                          <svg key={s} className="w-3 h-3" viewBox="0 0 20 20" fill={RATING_COLORS[i]}>
                            <path d="M10 1l2.39 4.84 5.34.78-3.87 3.77.91 5.33L10 13.28l-4.77 2.44.91-5.33L2.27 6.62l5.34-.78L10 1z" />
                          </svg>
                        ))}
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="flex h-32 items-center justify-center text-sm text-muted-foreground">
                {t('consciousness_statistics.no_data')}
              </div>
            )}
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_statistics.episodes_over_time')}</CardTitle>
              <CardDescription>{t('consciousness_statistics.episodes_over_time_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              {episodesByDay.counts.length > 0 ? (
                <BarChart
                  data={episodesByDay.counts}
                  labels={episodesByDay.labels}
                  colors={['#8b5cf6', '#6366f1', '#3b82f6', '#06b6d4', '#22c55e']}
                  yLabel="Episodes"
                  tooltip={tooltipBar}
                  onTooltip={setTooltipBar}
                />
              ) : (
                <div className="flex h-[300px] items-center justify-center text-sm text-muted-foreground">
                  {t('consciousness_statistics.no_data')}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_statistics.rating_over_time')}</CardTitle>
              <CardDescription>{t('consciousness_statistics.rating_over_time_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              {avgRatingByDay.values.length > 0 ? (
                <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full h-auto">
                  <GridLines yMin={0} yMax={5} steps={4} />
                  <XAxisLabels labels={avgRatingByDay.labels} />
                  <text x={M.left - 40} y={M.top + PH / 2} textAnchor="middle" fontSize="10" className="fill-muted-foreground" transform={`rotate(-90, ${M.left - 40}, ${M.top + PH / 2})`}>Rating</text>
                  <text x={M.left + PW / 2} y={SVG_H - 2} textAnchor="middle" fontSize="10" className="fill-muted-foreground">Day</text>
                  <path d={buildLinePath(avgRatingByDay.values, 0, 5)} fill="none" stroke="#8b5cf6" strokeWidth="2" />
                  {avgRatingByDay.values.map((v, i) => {
                    const xStep = avgRatingByDay.values.length > 1 ? PW / (avgRatingByDay.values.length - 1) : PW / 2
                    const x = M.left + (avgRatingByDay.values.length > 1 ? i * xStep : PW / 2)
                    const y = M.top + PH - (v / 5) * PH
                    return (
                      <circle key={i} cx={x} cy={y} r="3" fill="#8b5cf6" stroke="hsl(var(--background))" strokeWidth="1"
                        onMouseEnter={() => setTooltipLine1({ x, y, lines: [avgRatingByDay.labels[i], `Avg Rating: ${v.toFixed(2)}`] })}
                        onMouseLeave={() => setTooltipLine1(null)}
                      />
                    )
                  })}
                  {tooltipLine1 && <TooltipBox x={tooltipLine1.x} y={tooltipLine1.y} lines={tooltipLine1.lines} />}
                </svg>
              ) : (
                <div className="flex h-[300px] items-center justify-center text-sm text-muted-foreground">
                  {t('consciousness_statistics.no_data')}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_statistics.growth_trend')}</CardTitle>
            <CardDescription>{t('consciousness_statistics.growth_trend_desc')}</CardDescription>
          </CardHeader>
          <CardContent>
            {growthTrend.values.length > 0 ? (
              <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full h-auto">
                <GridLines yMin={-0.1} yMax={0.1} steps={4} />
                <XAxisLabels labels={growthTrend.labels} />
                <text x={M.left - 40} y={M.top + PH / 2} textAnchor="middle" fontSize="10" className="fill-muted-foreground" transform={`rotate(-90, ${M.left - 40}, ${M.top + PH / 2})`}>Growth Delta</text>
                <text x={M.left + PW / 2} y={SVG_H - 2} textAnchor="middle" fontSize="10" className="fill-muted-foreground">Episode</text>
                <line x1={M.left} y1={M.top + PH / 2} x2={M.left + PW} y2={M.top + PH / 2} stroke="hsl(var(--muted-foreground))" strokeWidth="0.5" strokeDasharray="4 2" />
                <path d={buildAreaPath(growthTrend.values.map((v, i) => v >= 0 ? v : 0), -0.1, 0.1, 0)} fill="#22c55e" fillOpacity="0.15" />
                <path d={buildAreaPath(growthTrend.values.map((v) => v <= 0 ? v : 0), -0.1, 0.1, 0)} fill="#ef4444" fillOpacity="0.15" />
                <path d={buildLinePath(growthTrend.values, -0.1, 0.1)} fill="none" stroke="hsl(var(--foreground))" strokeWidth="1.5" />
                {growthTrend.avg.some(v => v !== null) && (
                  <path d={buildLinePath(growthTrend.avg.map(v => v ?? 0), -0.1, 0.1)} fill="none" stroke="#8b5cf6" strokeWidth="2" strokeDasharray="6 3" />
                )}
                {growthTrend.values.map((v, i) => {
                  const xStep = growthTrend.values.length > 1 ? PW / (growthTrend.values.length - 1) : PW / 2
                  const x = M.left + (growthTrend.values.length > 1 ? i * xStep : PW / 2)
                  const y = M.top + PH - ((v - (-0.1)) / (0.1 - (-0.1))) * PH
                  return (
                    <circle key={i} cx={x} cy={y} r="3" fill={v >= 0 ? '#22c55e' : '#ef4444'} stroke="hsl(var(--background))" strokeWidth="1"
                      onMouseEnter={() => setTooltipLine2({ x, y, lines: [`Episode ${i + 1}`, `Growth: ${(v * 100).toFixed(2)}%`] })}
                      onMouseLeave={() => setTooltipLine2(null)}
                    />
                  )
                })}
                {tooltipLine2 && <TooltipBox x={tooltipLine2.x} y={tooltipLine2.y} lines={tooltipLine2.lines} />}
                <g>
                  <line x1={M.left + PW - 120} y1={M.top + 8} x2={M.left + PW - 100} y2={M.top + 8} stroke="hsl(var(--foreground))" strokeWidth="1.5" />
                  <text x={M.left + PW - 96} y={M.top + 12} fontSize="10" className="fill-muted-foreground">Raw</text>
                  <line x1={M.left + PW - 120} y1={M.top + 24} x2={M.left + PW - 100} y2={M.top + 24} stroke="#8b5cf6" strokeWidth="2" strokeDasharray="6 3" />
                  <text x={M.left + PW - 96} y={M.top + 28} fontSize="10" className="fill-muted-foreground">SMA(5)</text>
                </g>
              </svg>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-sm text-muted-foreground">
                {t('consciousness_statistics.no_data')}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_statistics.insights_title')}</CardTitle>
            <CardDescription>{t('consciousness_statistics.insights_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="text-sm font-medium mb-2">{t('consciousness_statistics.top_qualia')}</h4>
              <div className="flex flex-wrap gap-2">
                {topQualia.length > 0 ? topQualia.map(([key, count]) => (
                  <span key={key} className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs" style={{ backgroundColor: `${QUALIA_COLORS[key]}20`, color: QUALIA_COLORS[key] }}>
                    {key} ({count})
                  </span>
                )) : <span className="text-sm text-muted-foreground">{t('consciousness_statistics.no_data')}</span>}
              </div>
            </div>

            {(bestWorst.best || bestWorst.worst) && (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {bestWorst.best && (
                  <div>
                    <h4 className="text-sm font-medium mb-1">{t('consciousness_statistics.most_positive')}</h4>
                    <p className="text-xs text-muted-foreground line-clamp-2">{bestWorst.best.narrative || bestWorst.best.response}</p>
                    <span className="text-xs text-green-500">{t('consciousness_statistics.rating')}: {bestWorst.best.rating}/5</span>
                  </div>
                )}
                {bestWorst.worst && (
                  <div>
                    <h4 className="text-sm font-medium mb-1">{t('consciousness_statistics.most_negative')}</h4>
                    <p className="text-xs text-muted-foreground line-clamp-2">{bestWorst.worst.narrative || bestWorst.worst.response}</p>
                    <span className="text-xs text-red-500">{t('consciousness_statistics.rating')}: {bestWorst.worst.rating}/5</span>
                  </div>
                )}
              </div>
            )}

            {beliefTrends.length > 0 && (
              <div>
                <h4 className="text-sm font-medium mb-2">{t('consciousness_statistics.belief_trends')}</h4>
                <div className="space-y-1">
                  {beliefTrends.map(b => (
                    <div key={b.key} className="flex items-center gap-2 text-xs">
                      <span className={b.growing ? 'text-green-500' : 'text-red-500'}>{b.growing ? '↑' : '↓'}</span>
                      <span className="text-muted-foreground">{b.key}</span>
                      <span className={b.growing ? 'text-green-500' : 'text-red-500'}>
                        {b.delta >= 0 ? '+' : ''}{(b.delta * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex justify-end">
              <button onClick={handleExport} className="inline-flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors focus-visible:ring-2 focus-visible:ring-ring">
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                {t('consciousness_statistics.export_json')}
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
