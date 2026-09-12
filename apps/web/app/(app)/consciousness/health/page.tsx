'use client'

import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import {
  Card, CardContent, CardHeader, CardTitle, Skeleton, Switch,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'

interface HealthData {
  status: string
  health_score: number
  enabled: boolean
  level: number
  episodes: number
  avg_growth: number
  positive_ratio: number
  qualia: Record<string, number>
  last_reflection: string
}

interface StatusData {
  response_quality: number
  last_reflection: string
  training_pairs?: number
  [key: string]: unknown
}

interface EvalData {
  overall_score: number
  diagnostics: string[]
  metrics: Record<string, { score: number; weight: number; details: string }>
  timestamp?: string
  [key: string]: unknown
}

interface EpisodesData {
  episodes: Array<{ timestamp: number; growth_delta: number; [key: string]: unknown }>
  total: number
}

const HISTORY_KEY = 'consciousness_health_history'
const MAX_HISTORY = 20

function getScoreColor(score: number): string {
  if (score < 40) return '#ef4444'
  if (score < 70) return '#f59e0b'
  return '#22c55e'
}

function getScoreLabel(score: number, t: (k: string) => string): string {
  if (score < 40) return t('consciousness_health.status_critical')
  if (score < 70) return t('consciousness_health.status_warning')
  if (score < 90) return t('consciousness_health.status_healthy')
  return t('consciousness_health.status_excellent')
}

function formatTimeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function loadHistory(): Array<{ timestamp: number; score: number }> {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveHistory(history: Array<{ timestamp: number; score: number }>) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-MAX_HISTORY)))
  } catch (e) {
    console.error('Failed to save health history', e)
  }
}

export default function ConsciousnessHealthPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()
  const [loading, setLoading] = useState(true)
  const [health, setHealth] = useState<HealthData | null>(null)
  const [status, setStatus] = useState<StatusData | null>(null)
  const [evalData, setEvalData] = useState<EvalData | null>(null)
  const [episodesData, setEpisodesData] = useState<EpisodesData | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [lastUpdated, setLastUpdated] = useState<number>(Date.now() / 1000)
  const [history, setHistory] = useState<Array<{ timestamp: number; score: number }>>([])
  const prevScoreRef = useRef<number | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchAll = useCallback(async () => {
    try {
      const [hResult, sResult, eResult, epResult] = await Promise.allSettled([
        consciousnessController.healthCheck(),
        consciousnessController.getStatus(),
        consciousnessController.evaluate(),
        consciousnessController.getEpisodeHistory(50),
      ])

      if (hResult.status === 'fulfilled') {
        const data = hResult.value as unknown as HealthData
        setHealth(data)
        if (typeof data.health_score === 'number') {
          setHistory(prev => {
            const next = [...prev, { timestamp: Date.now() / 1000, score: data.health_score }].slice(-MAX_HISTORY)
            saveHistory(next)
            return next
          })
        }
      }
      if (sResult.status === 'fulfilled') {
        setStatus(sResult.value as unknown as StatusData)
      }
      if (eResult.status === 'fulfilled') {
        setEvalData(eResult.value as unknown as EvalData)
      }
      if (epResult.status === 'fulfilled') {
        setEpisodesData(epResult.value as unknown as EpisodesData)
      }
      setLastUpdated(Date.now() / 1000)
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    setHistory(loadHistory())
    fetchAll()
  }, [fetchAll])

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(fetchAll, 30000)
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [autoRefresh, fetchAll])

  const trend = useMemo(() => {
    if (history.length < 2) return 'stable'
    const recent = history.slice(-5)
    const firstHalf = recent.slice(0, Math.floor(recent.length / 2))
    const secondHalf = recent.slice(Math.floor(recent.length / 2))
    const avgFirst = firstHalf.reduce((s, h) => s + h.score, 0) / firstHalf.length
    const avgSecond = secondHalf.reduce((s, h) => s + h.score, 0) / secondHalf.length
    if (avgSecond > avgFirst + 2) return 'up'
    if (avgSecond < avgFirst - 2) return 'down'
    return 'stable'
  }, [history])

  const timelineData = useMemo(() => {
    const data = history.slice(-MAX_HISTORY)
    if (data.length === 0) return []
    return data.map((d, i) => ({
      x: data.length > 1 ? (i / (data.length - 1)) * 100 : 50,
      y: d.score,
      timestamp: d.timestamp,
    }))
  }, [history])

  const qualiaActive = useMemo(() => {
    if (!health?.qualia) return 0
    return Object.values(health.qualia).filter(v => v > 0.1).length
  }, [health?.qualia])

  const avgGrowthPct = health ? (health.avg_growth * 100).toFixed(1) : '—'
  const positiveRatioPct = health ? `${(health.positive_ratio * 100).toFixed(0)}%` : '—'
  const trainingPairs = (status as any)?.training?.total_pairs ?? (status as any)?.training_pairs ?? 0

  const diagnosticSeverity = (msg: string): 'info' | 'warning' | 'error' => {
    const lower = msg.toLowerCase()
    if (lower.includes('error') || lower.includes('fail') || lower.includes('critical')) return 'error'
    if (lower.includes('warn') || lower.includes('low') || lower.includes('decline')) return 'warning'
    return 'info'
  }

  const severityColor = (sev: string) => {
    if (sev === 'error') return 'text-red-500 bg-red-500/10'
    if (sev === 'warning') return 'text-amber-500 bg-amber-500/10'
    return 'text-blue-500 bg-blue-500/10'
  }

  if (loading) {
    return (
      <PageContainer title={t('consciousness_health.page_title')}>
        <div className="space-y-4 sm:space-y-6 p-4 sm:p-6">
          <Skeleton className="h-48 w-full max-w-xs mx-auto" />
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
          </div>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
          </div>
          <Skeleton className="h-48" />
          <Skeleton className="h-64" />
        </div>
      </PageContainer>
    )
  }

  const score = health?.health_score ?? 0
  const scoreColor = getScoreColor(score)
  const circumference = 2 * Math.PI * 54
  const fillAmount = (score / 100) * circumference

  return (
    <PageContainer title={t('consciousness_health.page_title')}>
      <div className="space-y-4 sm:space-y-6 p-4 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <Switch checked={autoRefresh} onCheckedChange={setAutoRefresh} />
            <span className="text-xs text-muted-foreground">
              {t('consciousness_health.auto_refresh')}
            </span>
          </div>
          <span className="text-xs text-muted-foreground">
            {t('consciousness_health.last_updated', { time: formatTimeAgo(lastUpdated) })}
          </span>
        </div>

        <Card>
          <CardContent className="p-4 sm:p-6 flex flex-col items-center">
            <svg viewBox="0 0 120 120" className="w-40 h-40 sm:w-56 sm:h-56">
              <circle cx="60" cy="60" r="54" fill="none" stroke="hsl(var(--border))" strokeWidth="8" />
              <circle
                cx="60" cy="60" r="54"
                fill="none"
                stroke={scoreColor}
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={circumference - fillAmount}
                transform="rotate(-90 60 60)"
                className="transition-all duration-1000 ease-out"
              />
              <text x="60" y="52" textAnchor="middle" className="fill-foreground" fontSize="24" fontWeight="bold">
                {score.toFixed(0)}
              </text>
              <text x="60" y="68" textAnchor="middle" className="fill-muted-foreground" fontSize="8">
                {t('consciousness_health.out_of_100')}
              </text>
              <text x="60" y="82" textAnchor="middle" fontSize="9" fontWeight="500" fill={scoreColor}>
                {getScoreLabel(score, t)}
              </text>
            </svg>
            <div className="mt-2 text-sm text-muted-foreground">
              {t('consciousness_health.trend')}: {' '}
              <span className={trend === 'up' ? 'text-green-500' : trend === 'down' ? 'text-red-500' : ''}>
                {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'} {t(`consciousness_health.trend_${trend}`)}
              </span>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-2 gap-2 sm:gap-3 md:grid-cols-5">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">{t('consciousness_health.enabled')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-lg font-bold ${health?.enabled ? 'text-green-500' : 'text-red-500'}`}>
                {health?.enabled ? t('consciousness_health.yes') : t('consciousness_health.no')}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">{t('consciousness_health.level')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-lg font-bold">{health?.level ?? '—'}/3</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">{t('consciousness_health.episodes')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-lg font-bold">{health?.episodes ?? episodesData?.total ?? '—'}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">{t('consciousness_health.training_pairs')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-lg font-bold">{trainingPairs}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">{t('consciousness_health.last_reflection')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xs text-muted-foreground truncate">
                {health?.last_reflection || status?.last_reflection || '—'}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-2 gap-2 sm:gap-3 md:grid-cols-3">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">{t('consciousness_health.avg_growth')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-xl font-bold ${health && health.avg_growth >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {health ? `${health.avg_growth >= 0 ? '+' : ''}${avgGrowthPct}%` : '—'}
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">{t('consciousness_health.positive_ratio')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xl font-bold">{positiveRatioPct}</div>
              <div className="h-2 mt-2 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${(health?.positive_ratio ?? 0) * 100}%`,
                    backgroundColor: scoreColor,
                  }}
                />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">{t('consciousness_health.qualia_diversity')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xl font-bold">{qualiaActive}/7</div>
              <div className="text-xs text-muted-foreground mt-1">{t('consciousness_health.dimensions_active')}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">{t('consciousness_health.belief_stability')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xl font-bold">
                {evalData?.metrics?.belief_stability?.score?.toFixed(0) ?? '—'}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">{t('consciousness_health.narrative_coherence')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xl font-bold">
                {evalData?.metrics?.narrative_coherence?.score?.toFixed(0) ?? '—'}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">{t('consciousness_health.feedback_alignment')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xl font-bold">
                {evalData?.metrics?.feedback_alignment?.score?.toFixed(0) ?? '—'}
              </div>
            </CardContent>
          </Card>
        </div>

        {evalData && evalData.diagnostics && evalData.diagnostics.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_health.diagnostics')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {evalData.diagnostics.map((msg: string, i: number) => {
                  const sev = diagnosticSeverity(msg)
                  return (
                    <div key={i} className={`flex items-start gap-2 rounded-md p-2 text-xs ${severityColor(sev)}`}>
                      <span className="shrink-0 mt-0.5">
                        {sev === 'error' ? '●' : sev === 'warning' ? '●' : '●'}
                      </span>
                      <span className="flex-1">{msg}</span>
                    </div>
                  )
                })}
              </div>
              {evalData.timestamp && (
                <div className="mt-3 text-xs text-muted-foreground">
                  {t('consciousness_health.last_evaluation')}: {new Date(evalData.timestamp).toLocaleString()}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {timelineData.length > 1 && (
          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_health.health_timeline')}</CardTitle>
            </CardHeader>
            <CardContent>
              <svg viewBox="0 0 400 150" className="w-full h-auto">
                <line x1="40" y1="10" x2="40" y2="130" stroke="hsl(var(--border))" strokeWidth="0.5" />
                <line x1="40" y1="130" x2="390" y2="130" stroke="hsl(var(--border))" strokeWidth="0.5" />
                {[0, 25, 50, 75, 100].map((v) => {
                  const y = 130 - (v / 100) * 120
                  return (
                    <g key={v}>
                      <line x1="40" y1={y} x2="390" y2={y} stroke="hsl(var(--border))" strokeWidth="0.5" strokeDasharray="3 3" />
                      <text x="35" y={y + 3} textAnchor="end" fontSize="8" className="fill-muted-foreground">{v}</text>
                    </g>
                  )
                })}
                {(() => {
                  const points = timelineData.map(d => {
                    const x = 40 + (d.x / 100) * 350
                    const y = 130 - (d.y / 100) * 120
                    return `${x.toFixed(1)},${y.toFixed(1)}`
                  }).join(' ')
                  return (
                    <polygon
                      points={`${40},${130} ${points} ${40 + ((timelineData[timelineData.length - 1]?.x ?? 0) / 100) * 350},130`}
                      fill="hsl(var(--primary))"
                      fillOpacity="0.08"
                    />
                  )
                })()}
                {(() => {
                  const pathParts = timelineData.map((d, i) => {
                    const x = 40 + (d.x / 100) * 350
                    const y = 130 - (d.y / 100) * 120
                    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
                  })
                  return (
                    <path d={pathParts.join(' ')} fill="none" stroke="hsl(var(--primary))" strokeWidth="2" />
                  )
                })()}
                {timelineData.map((d, i) => {
                  const x = 40 + (d.x / 100) * 350
                  const y = 130 - (d.y / 100) * 120
                  return (
                    <circle key={i} cx={x} cy={y} r="3" fill="hsl(var(--primary))" stroke="hsl(var(--background))" strokeWidth="1" />
                  )
                })}
              </svg>
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  )
}
