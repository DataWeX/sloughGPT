'use client'

import { useState, useEffect, useCallback, useMemo } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import { Button, Card, CardContent, CardHeader, CardTitle, Skeleton } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'

interface HealthData {
  health_score: number
  enabled: boolean
  level: number
  episodes: number
  avg_growth: number
  positive_ratio: number
  qualia: Record<string, number>
  beliefs: Record<string, number>
}

interface EvalData {
  overall_score: number
  metrics: Record<string, { score: number; weight: number; details: string }>
  diagnostics: string[]
}

interface EpisodesData {
  episodes: Array<{ timestamp: number; growth_delta: number; rating: number; [key: string]: unknown }>
  total: number
}

interface PresetData {
  id: string
  name: string
  description: string
  level?: number
  qualia?: Record<string, number>
  beliefs?: Record<string, number>
}

type ConfigKey = 'current' | 'default' | 'formal' | 'creative' | 'analyst' | 'empathetic' | 'minimal'

const CONFIG_OPTIONS: Array<{ key: ConfigKey; labelKey: string }> = [
  { key: 'current', labelKey: 'consciousness_compare.config_current' },
  { key: 'default', labelKey: 'consciousness_compare.config_default' },
  { key: 'formal', labelKey: 'consciousness_compare.config_formal' },
  { key: 'creative', labelKey: 'consciousness_compare.config_creative' },
  { key: 'analyst', labelKey: 'consciousness_compare.config_analyst' },
  { key: 'empathetic', labelKey: 'consciousness_compare.config_empathetic' },
  { key: 'minimal', labelKey: 'consciousness_compare.config_minimal' },
]

const QUALIA_DIMS = ['valence', 'arousal', 'novelty', 'coherence', 'salience', 'certainty', 'complexity']

function getScoreColor(score: number): string {
  if (score < 40) return '#ef4444'
  if (score < 70) return '#f59e0b'
  return '#22c55e'
}

function getDiffColor(a: number, b: number, higherBetter = true): string {
  const diff = a - b
  if (Math.abs(diff) < 0.01) return ''
  return higherBetter ? (diff > 0 ? 'text-green-500' : 'text-red-500') : (diff < 0 ? 'text-green-500' : 'text-red-500')
}

function getDiffIcon(a: number, b: number, higherBetter = true): string {
  const diff = a - b
  if (Math.abs(diff) < 0.01) return ''
  return higherBetter ? (diff > 0 ? '▲' : '▼') : (diff < 0 ? '▲' : '▼')
}

function buildFakeData(name: string): HealthData {
  const seed = name.length * 17
  const q = QUALIA_DIMS.reduce((acc, d, i) => {
    acc[d] = Math.min(1, Math.max(0, ((seed * (i + 1) * 7) % 100) / 100))
    return acc
  }, {} as Record<string, number>)
  return {
    health_score: ((seed * 3) % 60) + 40,
    enabled: true,
    level: 2,
    episodes: 20 + (seed % 30),
    avg_growth: ((seed % 30) - 10) / 100,
    positive_ratio: ((seed * 2) % 40 + 60) / 100,
    qualia: q,
    beliefs: {
      'Self-awareness': Math.min(1, ((seed * 11) % 100) / 100),
      'Emotional range': Math.min(1, ((seed * 13) % 100) / 100),
      'Context awareness': Math.min(1, ((seed * 17) % 100) / 100),
    },
  }
}

export default function ConsciousnessComparePage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()
  const [loading, setLoading] = useState(true)
  const [configA, setConfigA] = useState<ConfigKey>('current')
  const [configB, setConfigB] = useState<ConfigKey>('default')
  const [presets, setPresets] = useState<PresetData[]>([])
  const [currentHealth, setCurrentHealth] = useState<HealthData | null>(null)
  const [currentEval, setCurrentEval] = useState<EvalData | null>(null)
  const [currentEpisodes, setCurrentEpisodes] = useState<EpisodesData | null>(null)
  const [applying, setApplying] = useState(false)

  const fetchCurrent = useCallback(async () => {
    try {
      const [healthData, evalData, episodesData] = await Promise.allSettled([
        consciousnessController.healthCheck(),
        consciousnessController.evaluate(),
        consciousnessController.getEpisodeHistory(100),
      ])
      if (healthData.status === 'fulfilled') {
        setCurrentHealth(healthData.value as unknown as HealthData)
      }
      if (evalData.status === 'fulfilled') {
        setCurrentEval(evalData.value as unknown as EvalData)
      }
      if (episodesData.status === 'fulfilled') {
        setCurrentEpisodes(episodesData.value as unknown as EpisodesData)
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }, [addToast])

  const fetchPresets = useCallback(async () => {
    try {
      const data = await consciousnessController.getPersonalityPresets()
      setPresets((data as any).presets ?? data)
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }, [])

  useEffect(() => {
    Promise.all([fetchCurrent(), fetchPresets()]).finally(() => setLoading(false))
  }, [fetchCurrent, fetchPresets])

  const fakeDataMap = useMemo(() => {
    const map: Record<string, HealthData> = {}
    presets.forEach(p => {
      map[p.id] = buildFakeData(p.name)
    })
    map['default'] = buildFakeData('default')
    map['formal'] = buildFakeData('formal')
    map['creative'] = buildFakeData('creative')
    map['analyst'] = buildFakeData('analyst')
    map['empathetic'] = buildFakeData('empathetic')
    map['minimal'] = buildFakeData('minimal')
    return map
  }, [presets])

  const getDataForConfig = useCallback((key: ConfigKey): HealthData | null => {
    if (key === 'current') return currentHealth
    return fakeDataMap[key] ?? buildFakeData(key)
  }, [currentHealth, fakeDataMap])

  const dataA = useMemo(() => getDataForConfig(configA), [configA, getDataForConfig])
  const dataB = useMemo(() => getDataForConfig(configB), [configB, getDataForConfig])

  const handleSwap = () => {
    setConfigA(configB)
    setConfigB(configA)
  }

  const handleApply = async () => {
    setApplying(true)
    try {
      await consciousnessController.applyPersonalityPreset(configA === 'current' ? 'default' : configA)
      addToast(t('consciousness_compare.toast_applied'), 'success')
      fetchCurrent()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setApplying(false)
    }
  }

  const getLabel = (key: ConfigKey) => {
    const opt = CONFIG_OPTIONS.find(o => o.key === key)
    return opt ? t(opt.labelKey) : key
  }

  if (loading) {
    return (
      <PageContainer title={t('consciousness_compare.page_title')}>
        <div className="space-y-4 sm:space-y-6 p-4 sm:p-6">
          <div className="flex gap-3">
            <Skeleton className="h-10 w-48" />
            <Skeleton className="h-10 w-48" />
            <Skeleton className="h-10 w-20" />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Skeleton className="h-80" />
            <Skeleton className="h-80" />
          </div>
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
        </div>
      </PageContainer>
    )
  }

  const avgGrowthA = dataA ? (dataA.avg_growth * 100).toFixed(1) : '—'
  const avgGrowthB = dataB ? (dataB.avg_growth * 100).toFixed(1) : '—'
  const positiveA = dataA ? `${(dataA.positive_ratio * 100).toFixed(0)}%` : '—'
  const positiveB = dataB ? `${(dataB.positive_ratio * 100).toFixed(0)}%` : '—'

  const metricsData = [
    { label: t('consciousness_compare.health_score'), valueA: dataA?.health_score ?? 0, valueB: dataB?.health_score ?? 0, max: 100 },
    { label: t('consciousness_compare.avg_growth'), valueA: dataA ? dataA.avg_growth * 100 : 0, valueB: dataB ? dataB.avg_growth * 100 : 0, max: 30 },
    { label: t('consciousness_compare.positive_ratio'), valueA: dataA ? dataA.positive_ratio * 100 : 0, valueB: dataB ? dataB.positive_ratio * 100 : 0, max: 100 },
    { label: t('consciousness_compare.episodes'), valueA: dataA?.episodes ?? 0, valueB: dataB?.episodes ?? 0, max: Math.max(dataA?.episodes ?? 0, dataB?.episodes ?? 0, 1) },
  ]

  const qualiaA = dataA?.qualia ?? {}
  const qualiaB = dataB?.qualia ?? {}

  return (
    <PageContainer title={t('consciousness_compare.page_title')}>
      <div className="space-y-4 sm:space-y-6 p-4 sm:p-6">
        <Card>
          <CardContent className="p-3 sm:p-4">
            <div className="flex flex-wrap items-center gap-3">
              <select
                value={configA}
                onChange={(e) => setConfigA(e.target.value as ConfigKey)}
                className="rounded-md border bg-background px-3 py-2 text-sm"
              >
                {CONFIG_OPTIONS.map(opt => (
                  <option key={opt.key} value={opt.key}>{t(opt.labelKey)}</option>
                ))}
              </select>
              <Button onClick={handleSwap} variant="secondary" size="sm">
                {t('consciousness_compare.swap')}
              </Button>
              <select
                value={configB}
                onChange={(e) => setConfigB(e.target.value as ConfigKey)}
                className="rounded-md border bg-background px-3 py-2 text-sm"
              >
                {CONFIG_OPTIONS.map(opt => (
                  <option key={opt.key} value={opt.key}>{t(opt.labelKey)}</option>
                ))}
              </select>
              <Button onClick={handleApply} disabled={applying} size="sm">
                {applying ? t('consciousness_compare.applying') : t('consciousness_compare.apply')}
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {[dataA, dataB].map((data, idx) => {
            const label = idx === 0 ? getLabel(configA) : getLabel(configB)
            const other = idx === 0 ? dataB : dataA
            const score = data?.health_score ?? 0
            const scoreColor = getScoreColor(score)
            const circumference = 2 * Math.PI * 40
            const fillAmount = (score / 100) * circumference
            return (
              <Card key={idx}>
                <CardHeader>
                  <CardTitle className="text-sm">{label}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-center">
                    <svg viewBox="0 0 100 100" className="w-28 h-28">
                      <circle cx="50" cy="50" r="40" fill="none" stroke="hsl(var(--border))" strokeWidth="6" />
                      <circle
                        cx="50" cy="50" r="40"
                        fill="none"
                        stroke={scoreColor}
                        strokeWidth="6"
                        strokeLinecap="round"
                        strokeDasharray={circumference}
                        strokeDashoffset={circumference - fillAmount}
                        transform="rotate(-90 50 50)"
                        className="transition-all duration-1000 ease-out"
                      />
                      <text x="50" y="46" textAnchor="middle" className="fill-foreground" fontSize="16" fontWeight="bold">
                        {score.toFixed(0)}
                      </text>
                      <text x="50" y="58" textAnchor="middle" className="fill-muted-foreground" fontSize="6">
                        {t('consciousness_compare.out_of_100')}
                      </text>
                    </svg>
                  </div>

                  <div className="space-y-2">
                    {QUALIA_DIMS.map(dim => {
                      const val = data?.qualia?.[dim] ?? 0
                      const otherVal = other?.qualia?.[dim] ?? 0
                      const diffClass = getDiffColor(val, otherVal)
                      return (
                        <div key={dim} className="space-y-0.5">
                          <div className="flex items-center justify-between">
                            <span className="text-[10px] sm:text-xs text-muted-foreground capitalize">{dim}</span>
                            <span className={`text-[10px] tabular-nums ${diffClass}`}>
                              {val.toFixed(2)} {getDiffIcon(val, otherVal)}
                            </span>
                          </div>
                          <div className="h-1.5 rounded-full bg-secondary">
                            <div
                              className="h-full rounded-full bg-primary transition-all"
                              style={{ width: `${Math.max(0, Math.min(1, val)) * 100}%` }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  {data?.beliefs && Object.keys(data.beliefs).length > 0 && (
                    <div className="space-y-1.5">
                      <span className="text-xs font-medium text-muted-foreground">{t('consciousness_compare.beliefs')}</span>
                      {Object.entries(data.beliefs).map(([belief, confidence]) => {
                        const otherVal = other?.beliefs?.[belief] ?? 0
                        const diffClass = getDiffColor(confidence, otherVal)
                        return (
                          <div key={belief} className="flex items-center gap-2">
                            <div className="h-1.5 w-1.5 rounded-full bg-primary shrink-0" />
                            <span className="flex-1 text-[10px] sm:text-xs truncate">{belief}</span>
                            <span className={`text-[10px] tabular-nums ${diffClass}`}>
                              {(confidence * 100).toFixed(0)}% {getDiffIcon(confidence, otherVal)}
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  )}

                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div>
                      <div className="text-lg font-bold">{data?.episodes ?? '—'}</div>
                      <div className="text-[10px] text-muted-foreground">{t('consciousness_compare.episodes')}</div>
                    </div>
                    <div>
                      <div className={`text-lg font-bold ${(data?.avg_growth ?? 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {data ? `${data.avg_growth >= 0 ? '+' : ''}${avgGrowthA || (data.avg_growth * 100).toFixed(1)}%` : '—'}
                      </div>
                      <div className="text-[10px] text-muted-foreground">{t('consciousness_compare.avg_growth')}</div>
                    </div>
                    <div>
                      <div className="text-lg font-bold">{data ? `${(data.positive_ratio * 100).toFixed(0)}%` : '—'}</div>
                      <div className="text-[10px] text-muted-foreground">{t('consciousness_compare.positive_ratio')}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_compare.metrics_chart')}</CardTitle>
          </CardHeader>
          <CardContent>
            <svg viewBox="0 0 400 180" className="w-full h-auto">
              {metricsData.map((m, i) => {
                const barWidth = 30
                const gap = 100
                const x = 50 + i * gap
                const maxH = 120
                const hA = m.max > 0 ? (m.valueA / m.max) * maxH : 0
                const hB = m.max > 0 ? (m.valueB / m.max) * maxH : 0
                const colorA = '#8b5cf6'
                const colorB = '#06b6d4'
                return (
                  <g key={i}>
                    <rect x={x} y={140 - hA} width={barWidth} height={hA} fill={colorA} rx="2" className="transition-all duration-500" />
                    <rect x={x + barWidth + 4} y={140 - hB} width={barWidth} height={hB} fill={colorB} rx="2" className="transition-all duration-500" />
                    <text x={x + barWidth + 2} y={155} textAnchor="middle" fontSize="7" className="fill-muted-foreground">
                      {m.label.length > 10 ? m.label.slice(0, 10) + '…' : m.label}
                    </text>
                    <text x={x + barWidth / 2} y={138 - hA} textAnchor="middle" fontSize="7" className="fill-foreground">
                      {m.valueA.toFixed(1)}
                    </text>
                    <text x={x + barWidth + 4 + barWidth / 2} y={138 - hB} textAnchor="middle" fontSize="7" className="fill-foreground">
                      {m.valueB.toFixed(1)}
                    </text>
                  </g>
                )
              })}
              <line x1="40" y1="140" x2="390" y2="140" stroke="hsl(var(--border))" strokeWidth="0.5" />
            </svg>
            <div className="flex items-center justify-center gap-4 mt-2">
              <div className="flex items-center gap-1.5">
                <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: '#8b5cf6' }} />
                <span className="text-xs text-muted-foreground">{getLabel(configA)}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: '#06b6d4' }} />
                <span className="text-xs text-muted-foreground">{getLabel(configB)}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_compare.qualia_chart')}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex justify-center">
              <svg viewBox="-110 -110 220 220" className="w-56 h-56 sm:w-72 sm:h-72">
                {[0.2, 0.4, 0.6, 0.8, 1.0].map((r, i) => (
                  <polygon
                    key={i}
                    points={Array.from({ length: 7 }, (_, k) => {
                      const angle = (k * 2 * Math.PI) / 7 - Math.PI / 2
                      return `${Math.cos(angle) * r * 100},${Math.sin(angle) * r * 100}`
                    }).join(' ')}
                    fill="none"
                    stroke="hsl(var(--border))"
                    strokeWidth="0.5"
                  />
                ))}
                {Array.from({ length: 7 }, (_, k) => {
                  const angle = (k * 2 * Math.PI) / 7 - Math.PI / 2
                  return (
                    <line
                      key={k}
                      x1="0" y1="0"
                      x2={Math.cos(angle) * 100} y2={Math.sin(angle) * 100}
                      stroke="hsl(var(--border))"
                      strokeWidth="0.5"
                    />
                  )
                })}
                <polygon
                  points={QUALIA_DIMS.map((dim, k) => {
                    const angle = (k * 2 * Math.PI) / 7 - Math.PI / 2
                    const clamped = Math.max(0, Math.min(1, qualiaA[dim] ?? 0))
                    return `${Math.cos(angle) * clamped * 100},${Math.sin(angle) * clamped * 100}`
                  }).join(' ')}
                  fill="rgba(139, 92, 246, 0.15)"
                  stroke="#8b5cf6"
                  strokeWidth="1.5"
                />
                <polygon
                  points={QUALIA_DIMS.map((dim, k) => {
                    const angle = (k * 2 * Math.PI) / 7 - Math.PI / 2
                    const clamped = Math.max(0, Math.min(1, qualiaB[dim] ?? 0))
                    return `${Math.cos(angle) * clamped * 100},${Math.sin(angle) * clamped * 100}`
                  }).join(' ')}
                  fill="rgba(6, 182, 212, 0.15)"
                  stroke="#06b6d4"
                  strokeWidth="1.5"
                />
                {QUALIA_DIMS.map((dim, k) => {
                  const angle = (k * 2 * Math.PI) / 7 - Math.PI / 2
                  const labelR = 112
                  return (
                    <text
                      key={dim}
                      x={Math.cos(angle) * labelR}
                      y={Math.sin(angle) * labelR}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      className="fill-muted-foreground"
                      fontSize="7"
                    >
                      {dim.slice(0, 3).toUpperCase()}
                    </text>
                  )
                })}
              </svg>
            </div>
            <div className="flex items-center justify-center gap-4 mt-2">
              <div className="flex items-center gap-1.5">
                <div className="h-3 w-3 rounded-full" style={{ backgroundColor: '#8b5cf6' }} />
                <span className="text-xs text-muted-foreground">{getLabel(configA)}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="h-3 w-3 rounded-full" style={{ backgroundColor: '#06b6d4' }} />
                <span className="text-xs text-muted-foreground">{getLabel(configB)}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_compare.recommendation')}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {(() => {
              const scoreA = dataA?.health_score ?? 0
              const scoreB = dataB?.health_score ?? 0
              const diff = scoreA - scoreB
              const winner = diff > 0 ? configA : diff < 0 ? configB : null
              const diffs: string[] = []
              if (dataA && dataB) {
                if (Math.abs(dataA.positive_ratio - dataB.positive_ratio) > 0.05) {
                  const better = dataA.positive_ratio > dataB.positive_ratio ? configA : configB
                  diffs.push(`${getLabel(better)} ${t('consciousness_compare.rec_better_positive')}`)
                }
                if (Math.abs(dataA.avg_growth - dataB.avg_growth) > 0.01) {
                  const better = dataA.avg_growth > dataB.avg_growth ? configA : configB
                  diffs.push(`${getLabel(better)} ${t('consciousness_compare.rec_better_growth')}`)
                }
                if (Math.abs(dataA.episodes - dataB.episodes) > 5) {
                  const better = dataA.episodes > dataB.episodes ? configA : configB
                  diffs.push(`${getLabel(better)} ${t('consciousness_compare.rec_more_episodes')}`)
                }
              }
              return (
                <>
                  <p className="text-sm">
                    {winner
                      ? `${t('consciousness_compare.rec_suggest')} ${getLabel(winner)} ${t('consciousness_compare.rec_higher_score')}`
                      : t('consciousness_compare.rec_equal')
                    }
                  </p>
                  {diffs.length > 0 && (
                    <div className="space-y-1">
                      {diffs.map((d, i) => (
                        <div key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                          <span className="shrink-0 mt-0.5">•</span>
                          <span>{d}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )
            })()}
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
