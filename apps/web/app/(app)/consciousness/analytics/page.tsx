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

interface QualiaPoint {
  valence: number
  arousal: number
  novelty: number
  coherence: number
  salience: number
  certainty: number
  complexity: number
  timestamp: string
}

interface BeliefsData {
  labels: string[]
  competence: number[]
  helpfulness: number[]
  creativity: number[]
  accuracy: number[]
  empathy: number[]
}

interface PersonalityHistoryPoint {
  timestamp: string
  voice: Record<string, unknown>
  traits: Record<string, unknown>
}

interface PersonalityHistory {
  history: PersonalityHistoryPoint[]
}

const QUALIA_DIMS = [
  { key: 'valence', color: '#8b5cf6' },
  { key: 'arousal', color: '#ef4444' },
  { key: 'novelty', color: '#f59e0b' },
  { key: 'coherence', color: '#22c55e' },
  { key: 'salience', color: '#3b82f6' },
  { key: 'certainty', color: '#06b6d4' },
  { key: 'complexity', color: '#d946ef' },
]

const BELIEF_KEYS = ['competence', 'helpfulness', 'creativity', 'accuracy', 'empathy'] as const
const BELIEF_COLORS: Record<string, string> = {
  competence: '#6366f1',
  helpfulness: '#22c55e',
  creativity: '#f59e0b',
  accuracy: '#3b82f6',
  empathy: '#ec4899',
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
  const boxW = 140
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

export default function ConsciousnessAnalyticsPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()
  const [loading, setLoading] = useState(true)
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [qualiaHistory, setQualiaHistory] = useState<QualiaPoint[]>([])
  const [beliefsData, setBeliefsData] = useState<BeliefsData | null>(null)
  const [personalityHistory, setPersonalityHistory] = useState<PersonalityHistory | null>(null)

  const [tooltip1, setTooltip1] = useState<{ x: number; y: number; lines: string[] } | null>(null)
  const [tooltip2, setTooltip2] = useState<{ x: number; y: number; lines: string[] } | null>(null)
  const [tooltip3, setTooltip3] = useState<{ x: number; y: number; lines: string[] } | null>(null)
  const [tooltip4, setTooltip4] = useState<{ x: number; y: number; lines: string[] } | null>(null)

  const fetchAll = useCallback(async () => {
    try {
      const [episodesResult, qualiaResult, beliefsResult] = await Promise.allSettled([
        consciousnessController.getEpisodeHistory(100),
        consciousnessController.getQualiaHistory(100),
        consciousnessController.getBeliefsHistory(),
      ])

      if (episodesResult.status === 'fulfilled') {
        setEpisodes((episodesResult.value as unknown as { episodes: Episode[] })?.episodes ?? [])
      }
      if (qualiaResult.status === 'fulfilled') {
        setQualiaHistory((qualiaResult.value as unknown as { qualia: QualiaPoint[] })?.qualia ?? [])
      }
      if (beliefsResult.status === 'fulfilled') {
        setBeliefsData((beliefsResult.value as unknown as { beliefs: BeliefsData })?.beliefs ?? null)
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { fetchAll() }, [fetchAll])

  const stats = useMemo(() => {
    if (episodes.length === 0) return null
    const growths = episodes.map(e => e.growth_delta)
    const ratings = episodes.map(e => e.rating).filter(r => r > 0)
    const avgGrowth = growths.reduce((a, b) => a + b, 0) / growths.length
    const avgRating = ratings.length > 0 ? ratings.reduce((a, b) => a + b, 0) / ratings.length : 0
    const positiveRatio = growths.filter(g => g > 0).length / growths.length
    const half = Math.floor(growths.length / 2)
    const firstHalf = growths.slice(0, half).reduce((a, b) => a + b, 0) / (half || 1)
    const secondHalf = growths.slice(half).reduce((a, b) => a + b, 0) / ((growths.length - half) || 1)
    const trend = secondHalf > firstHalf + 0.005 ? 'improving' : secondHalf < firstHalf - 0.005 ? 'declining' : 'stable'
    return {
      total: episodes.length,
      avgGrowth,
      avgRating,
      positiveRatio,
      trend,
      totalEpisodes: episodes.length,
    }
  }, [episodes])

  const growthData = useMemo(() => {
    const values = episodes.map(e => e.growth_delta)
    const avg = movingAverage(values, 5)
    const labels = episodes.map((_, i) => `${i + 1}`)
    return { values, avg, labels }
  }, [episodes])

  const ratingDist = useMemo(() => {
    const counts = [0, 0, 0, 0, 0]
    episodes.forEach(e => { if (e.rating >= 1 && e.rating <= 5) counts[e.rating - 1]++ })
    return counts
  }, [episodes])

  if (loading) {
    return (
      <PageContainer title={t('consciousness_analytics.page_title')}>
        <div className="space-y-6 p-6">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
          </div>
          <Skeleton className="h-72" />
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
    <PageContainer title={t('consciousness_analytics.page_title')}>
      <div className="space-y-6 p-6">
        {stats && (
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">{t('consciousness_analytics.total_episodes')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats.totalEpisodes}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">{t('consciousness_analytics.avg_growth')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold ${stats.avgGrowth >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {stats.avgGrowth >= 0 ? '+' : ''}{(stats.avgGrowth * 100).toFixed(1)}%
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">{t('consciousness_analytics.avg_rating')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats.avgRating.toFixed(1)} / 5</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">{t('consciousness_analytics.positive_ratio')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{(stats.positiveRatio * 100).toFixed(0)}%</div>
              </CardContent>
            </Card>
          </div>
        )}

        {stats && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground">{t('consciousness_analytics.trend')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-lg font-semibold ${
                stats.trend === 'improving' ? 'text-green-500' : stats.trend === 'declining' ? 'text-red-500' : 'text-muted-foreground'
              }`}>
                {stats.trend === 'improving' ? `↑ ${t('consciousness_analytics.trend_improving')}` :
                 stats.trend === 'declining' ? `↓ ${t('consciousness_analytics.trend_declining')}` :
                 `→ ${t('consciousness_analytics.trend_stable')}`}
              </div>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_analytics.growth_title')}</CardTitle>
            <CardDescription>{t('consciousness_analytics.growth_desc')}</CardDescription>
          </CardHeader>
          <CardContent>
            {growthData.values.length > 0 ? (
              <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full h-auto">
                <GridLines yMin={-0.1} yMax={0.1} steps={4} />
                <XAxisLabels labels={growthData.labels} />
                <text x={M.left - 40} y={M.top + PH / 2} textAnchor="middle" fontSize="10" className="fill-muted-foreground" transform={`rotate(-90, ${M.left - 40}, ${M.top + PH / 2})`}>Growth Delta</text>
                <text x={M.left + PW / 2} y={SVG_H - 2} textAnchor="middle" fontSize="10" className="fill-muted-foreground">Episode Index</text>
                <line x1={M.left} y1={M.top + PH / 2} x2={M.left + PW} y2={M.top + PH / 2} stroke="hsl(var(--muted-foreground))" strokeWidth="0.5" strokeDasharray="4 2" />
                <path d={buildAreaPath(growthData.values.map((v, i) => v >= 0 ? v : 0), -0.1, 0.1, 0)} fill="#22c55e" fillOpacity="0.15" />
                <path d={buildAreaPath(growthData.values.map((v) => v <= 0 ? v : 0), -0.1, 0.1, 0)} fill="#ef4444" fillOpacity="0.15" />
                <path d={buildLinePath(growthData.values, -0.1, 0.1)} fill="none" stroke="hsl(var(--foreground))" strokeWidth="1.5" />
                {growthData.avg.some(v => v !== null) && (
                  <path d={buildLinePath(growthData.avg.map(v => v ?? 0), -0.1, 0.1)} fill="none" stroke="#8b5cf6" strokeWidth="2" strokeDasharray="6 3" />
                )}
                {growthData.values.map((v, i) => {
                  const xStep = growthData.values.length > 1 ? PW / (growthData.values.length - 1) : PW / 2
                  const x = M.left + (growthData.values.length > 1 ? i * xStep : PW / 2)
                  const y = M.top + PH - ((v - (-0.1)) / (0.1 - (-0.1))) * PH
                  return (
                    <circle key={i} cx={x} cy={y} r="3" fill={v >= 0 ? '#22c55e' : '#ef4444'} stroke="hsl(var(--background))" strokeWidth="1"
                      onMouseEnter={() => setTooltip1({ x, y, lines: [`Episode ${i + 1}`, `Growth: ${(v * 100).toFixed(2)}%`] })}
                      onMouseLeave={() => setTooltip1(null)}
                    />
                  )
                })}
                {tooltip1 && <TooltipBox x={tooltip1.x} y={tooltip1.y} lines={tooltip1.lines} />}
                <g>
                  <line x1={M.left + PW - 120} y1={M.top + 8} x2={M.left + PW - 100} y2={M.top + 8} stroke="hsl(var(--foreground))" strokeWidth="1.5" />
                  <text x={M.left + PW - 96} y={M.top + 12} fontSize="10" className="fill-muted-foreground">Raw</text>
                  <line x1={M.left + PW - 120} y1={M.top + 24} x2={M.left + PW - 100} y2={M.top + 24} stroke="#8b5cf6" strokeWidth="2" strokeDasharray="6 3" />
                  <text x={M.left + PW - 96} y={M.top + 28} fontSize="10" className="fill-muted-foreground">SMA(5)</text>
                </g>
              </svg>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-sm text-muted-foreground">
                {t('consciousness_analytics.no_data')}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_analytics.qualia_title')}</CardTitle>
            <CardDescription>{t('consciousness_analytics.qualia_desc')}</CardDescription>
          </CardHeader>
          <CardContent>
            {qualiaHistory.length > 0 ? (
              <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full h-auto">
                <GridLines yMin={0} yMax={1} steps={4} />
                <XAxisLabels labels={qualiaHistory.map((_, i) => `${i + 1}`)} />
                <text x={M.left - 40} y={M.top + PH / 2} textAnchor="middle" fontSize="10" className="fill-muted-foreground" transform={`rotate(-90, ${M.left - 40}, ${M.top + PH / 2})`}>Value</text>
                <text x={M.left + PW / 2} y={SVG_H - 2} textAnchor="middle" fontSize="10" className="fill-muted-foreground">Qualia Index</text>
                {QUALIA_DIMS.map(({ key, color }) => {
                  const values = qualiaHistory.map(q => clamp(q[key as keyof QualiaPoint] as number, 0, 1))
                  return (
                    <path key={key} d={buildLinePath(values, 0, 1)} fill="none" stroke={color} strokeWidth="1.5" />
                  )
                })}
                {QUALIA_DIMS.map(({ key, color }) => {
                  const values = qualiaHistory.map(q => clamp(q[key as keyof QualiaPoint] as number, 0, 1))
                  return values.map((v, i) => {
                    const xStep = qualiaHistory.length > 1 ? PW / (qualiaHistory.length - 1) : PW / 2
                    const x = M.left + (qualiaHistory.length > 1 ? i * xStep : PW / 2)
                    const y = M.top + PH - v * PH
                    return (
                      <circle key={`${key}-${i}`} cx={x} cy={y} r="2.5" fill={color} stroke="hsl(var(--background))" strokeWidth="0.5"
                        onMouseEnter={() => setTooltip2({ x, y, lines: [`${key}: ${v.toFixed(3)}`, `Point ${i + 1}`] })}
                        onMouseLeave={() => setTooltip2(null)}
                      />
                    )
                  })
                })}
                {tooltip2 && <TooltipBox x={tooltip2.x} y={tooltip2.y} lines={tooltip2.lines} />}
                <g>
                  {QUALIA_DIMS.map(({ key, color }, i) => (
                    <g key={key} transform={`translate(${M.left + 8}, ${M.top + 8 + i * 16})`}>
                      <line x1="0" y1="0" x2="14" y2="0" stroke={color} strokeWidth="2" />
                      <text x="18" y="4" fontSize="9" className="fill-muted-foreground">{key}</text>
                    </g>
                  ))}
                </g>
              </svg>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-sm text-muted-foreground">
                {t('consciousness_analytics.no_data')}
              </div>
            )}
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_analytics.rating_title')}</CardTitle>
              <CardDescription>{t('consciousness_analytics.rating_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              {episodes.length > 0 ? (
                <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full h-auto">
                  <GridLines yMin={0} yMax={Math.max(1, ...ratingDist)} steps={4} />
                  {ratingDist.map((count, i) => {
                    const barW = PW / 7
                    const x = M.left + (PW / 5) * i + barW / 2
                    const barH = (count / Math.max(1, ...ratingDist)) * PH
                    const y = M.top + PH - barH
                    return (
                      <g key={i}>
                        <rect x={x} y={y} width={barW} height={barH} rx="3" fill={RATING_COLORS[i]} fillOpacity="0.8"
                          onMouseEnter={() => setTooltip3({ x: x + barW / 2, y, lines: [`Rating ${i + 1}: ${count} episodes`] })}
                          onMouseLeave={() => setTooltip3(null)}
                        />
                        <text x={x + barW / 2} y={SVG_H - 8} textAnchor="middle" fontSize="11" className="fill-muted-foreground">{i + 1}</text>
                        {count > 0 && (
                          <text x={x + barW / 2} y={y - 6} textAnchor="middle" fontSize="10" className="fill-foreground">{count}</text>
                        )}
                      </g>
                    )
                  })}
                  {tooltip3 && <TooltipBox x={tooltip3.x} y={tooltip3.y} lines={tooltip3.lines} />}
                </svg>
              ) : (
                <div className="flex h-[300px] items-center justify-center text-sm text-muted-foreground">
                  {t('consciousness_analytics.no_data')}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_analytics.beliefs_title')}</CardTitle>
              <CardDescription>{t('consciousness_analytics.beliefs_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              {beliefsData && beliefsData.labels.length > 1 ? (
                <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full h-auto">
                  <GridLines yMin={0} yMax={1} steps={4} />
                  <XAxisLabels labels={beliefsData.labels.map((_, i) => `${i + 1}`)} />
                  <text x={M.left - 40} y={M.top + PH / 2} textAnchor="middle" fontSize="10" className="fill-muted-foreground" transform={`rotate(-90, ${M.left - 40}, ${M.top + PH / 2})`}>Confidence</text>
                  <text x={M.left + PW / 2} y={SVG_H - 2} textAnchor="middle" fontSize="10" className="fill-muted-foreground">Step Index</text>
                  {BELIEF_KEYS.map((key) => {
                    const values = (beliefsData[key] as number[]) || []
                    return (
                      <path key={key} d={buildAreaPath(values, 0, 1, 0)} fill={BELIEF_COLORS[key]} fillOpacity="0.12" stroke={BELIEF_COLORS[key]} strokeWidth="1.5" />
                    )
                  })}
                  {BELIEF_KEYS.map((key) => {
                    const values = (beliefsData[key] as number[]) || []
                    return values.map((v, i) => {
                      const xStep = beliefsData.labels.length > 1 ? PW / (beliefsData.labels.length - 1) : PW / 2
                      const x = M.left + (beliefsData.labels.length > 1 ? i * xStep : PW / 2)
                      const y = M.top + PH - clamp(v, 0, 1) * PH
                      return (
                        <circle key={`${key}-${i}`} cx={x} cy={y} r="2" fill={BELIEF_COLORS[key]} stroke="hsl(var(--background))" strokeWidth="0.5"
                          onMouseEnter={() => setTooltip4({ x, y, lines: [`${key}: ${(v * 100).toFixed(1)}%`, `Step ${i + 1}`] })}
                          onMouseLeave={() => setTooltip4(null)}
                        />
                      )
                    })
                  })}
                  {tooltip4 && <TooltipBox x={tooltip4.x} y={tooltip4.y} lines={tooltip4.lines} />}
                  <g>
                    {BELIEF_KEYS.map((key, i) => (
                      <g key={key} transform={`translate(${M.left + 8}, ${M.top + 8 + i * 16})`}>
                        <line x1="0" y1="0" x2="14" y2="0" stroke={BELIEF_COLORS[key]} strokeWidth="2" />
                        <text x="18" y="4" fontSize="9" className="fill-muted-foreground">{key}</text>
                      </g>
                    ))}
                  </g>
                </svg>
              ) : (
                <div className="flex h-[300px] items-center justify-center text-sm text-muted-foreground">
                  {t('consciousness_analytics.no_data')}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </PageContainer>
  )
}
