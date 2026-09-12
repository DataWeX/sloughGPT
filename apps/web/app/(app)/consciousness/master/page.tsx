'use client'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Skeleton, Switch,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useConsciousnessStats } from '@/hooks/useConsciousnessStats'
import { useConsciousnessLive } from '@/hooks/useConsciousnessLive'
import { useLocale } from '@/hooks/useLocale'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, AreaChart, Area,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts'

const BELIEF_COLORS: Record<string, string> = {
  competence: '#6366f1',
  helpfulness: '#22c55e',
  creativity: '#f59e0b',
  accuracy: '#3b82f6',
  empathy: '#ec4899',
}

const QUALIA_DIMENSIONS = ['valence', 'arousal', 'novelty', 'coherence', 'salience', 'certainty', 'complexity']

function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function formatTimeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

export default function ConsciousnessMasterDashboardPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()
  const { stats, loading: statsLoading } = useConsciousnessStats()
  const { isLive, lastUpdate, toggleLive, latestEvent } = useConsciousnessLive()
  const [loading, setLoading] = useState(true)
  const [health, setHealth] = useState<any>(null)
  const [status, setStatus] = useState<any>(null)
  const [evalReport, setEvalReport] = useState<any>(null)
  const [episodes, setEpisodes] = useState<any[]>([])
  const [qualiaHistory, setQualiaHistory] = useState<any[]>([])
  const [beliefsHistory, setBeliefsHistory] = useState<any[]>([])
  const [currentBeliefs, setCurrentBeliefs] = useState<Record<string, number>>({})
  const [currentQualia, setCurrentQualia] = useState<Record<string, number>>({})
  const [seeding, setSeeding] = useState(false)
  const [reflecting, setReflecting] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)

  const fetchAll = useCallback(async () => {
    try {
      const [healthResult, statusResult, evalResult, epResult, qResult, bResult] = await Promise.allSettled([
        consciousnessController.healthCheck(),
        consciousnessController.getStatus(),
        consciousnessController.evaluate(),
        consciousnessController.getEpisodeHistory(50),
        consciousnessController.getQualiaHistory(100),
        consciousnessController.getBeliefsHistory(),
      ])

      if (healthResult.status === 'fulfilled') {
        setHealth(healthResult.value)
      }
      if (statusResult.status === 'fulfilled') {
        setStatus(statusResult.value)
        if (statusResult.value.beliefs) setCurrentBeliefs(statusResult.value.beliefs)
      }
      if (evalResult.status === 'fulfilled') {
        setEvalReport(evalResult.value)
      }
      if (epResult.status === 'fulfilled') {
        setEpisodes(epResult.value.episodes ?? [])
      }
      if (qResult.status === 'fulfilled') {
        setQualiaHistory(qResult.value.history ?? [])
      }
      if (bResult.status === 'fulfilled') {
        setBeliefsHistory(bResult.value.beliefs ?? [])
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { fetchAll() }, [fetchAll])

  useEffect(() => {
    if (!autoRefresh) return
    const id = setInterval(fetchAll, 30000)
    return () => clearInterval(id)
  }, [autoRefresh, fetchAll])

  useEffect(() => {
    if (!isLive || !latestEvent) return
    if (latestEvent.qualia) setCurrentQualia(latestEvent.qualia)
    if (latestEvent.beliefs) setCurrentBeliefs(latestEvent.beliefs as any)
  }, [latestEvent, isLive])

  const handleSeed = async () => {
    setSeeding(true)
    try {
      await consciousnessController.seedData({ count: 30 })
      addToast('30 episodes added', 'success')
      await fetchAll()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setSeeding(false)
    }
  }

  const handleReflect = async () => {
    setReflecting(true)
    try {
      await consciousnessController.reflect()
      addToast('New episode recorded', 'success')
      await fetchAll()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setReflecting(false)
    }
  }

  const handleExport = async () => {
    setExporting(true)
    try {
      const data = {
        health, status, evalReport, episodes, qualiaHistory, beliefsHistory,
        exportedAt: new Date().toISOString(),
      }
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `consciousness-master-${Date.now()}.json`
      a.click()
      URL.revokeObjectURL(url)
      addToast('Exported master dashboard data', 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setExporting(false)
    }
  }

  const totalEpisodes = stats?.total_episodes ?? episodes.length
  const avgGrowth = episodes.length > 0
    ? episodes.reduce((s: number, e: any) => s + (e.growth_delta ?? 0), 0) / episodes.length
    : 0
  const avgRating = episodes.length > 0
    ? episodes.reduce((s: number, e: any) => s + (e.rating ?? 0), 0) / episodes.length
    : 0
  const healthScore = health?.score ?? health?.health_score ?? 0
  const diagnostics = evalReport?.diagnostics ?? health?.diagnostics ?? []
  const lastReflection = episodes[episodes.length - 1]?.self_insight ?? null

  const qualiaRadarData = QUALIA_DIMENSIONS.map(dim => ({
    dimension: dim,
    value: currentQualia[dim] ?? 0,
    fullMark: 1,
  }))

  const beliefsBarData = Object.entries(currentBeliefs).map(([key, value]) => ({
    name: key,
    value: Math.max(0, Math.min(100, (value as number) * 100)),
    color: BELIEF_COLORS[key] || '#6366f1',
  }))

  const ratingDist = [1, 2, 3, 4, 5].map(r => ({
    rating: `${r}★`,
    count: episodes.filter((e: any) => Math.round(e.rating ?? 0) === r).length,
  }))

  const growthData = episodes.map((ep: any, i: number) => ({
    time: formatTime(ep.timestamp),
    growth: ep.growth_delta ?? 0,
    ma: episodes.slice(Math.max(0, i - 2), i + 1)
      .reduce((s: number, e: any) => s + (e.growth_delta ?? 0), 0) / Math.min(i + 1, 3),
  }))

  if (loading || statsLoading) {
    return (
      <PageContainer title="Consciousness Master Dashboard">
        <div className="space-y-4 sm:space-y-6 p-4 sm:p-6">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Skeleton className="h-28" />
            <Skeleton className="h-28" />
            <Skeleton className="h-28" />
            <Skeleton className="h-28" />
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <Skeleton className="h-64" />
            <Skeleton className="h-64" />
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <Skeleton className="h-48" />
            <Skeleton className="h-48" />
            <Skeleton className="h-48" />
          </div>
        </div>
      </PageContainer>
    )
  }

  const healthColor = healthScore > 80 ? '#22c55e' : healthScore > 50 ? '#f59e0b' : '#ef4444'

  return (
    <PageContainer
      title={t('consciousness_master.page_title')}
      subtitle={t('consciousness_master.subtitle')}
      headerRight={
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Switch checked={autoRefresh} onCheckedChange={setAutoRefresh} />
            <span className="text-xs text-muted-foreground">{t('consciousness_master.auto_refresh')}</span>
          </div>
          <Badge variant="outline" className="text-xs">
            {lastUpdate ? `Updated ${formatTimeAgo(lastUpdate / 1000)}` : t('consciousness_master.no_updates')}
          </Badge>
          {isLive && (
            <Badge variant="default" className="text-[10px] px-1.5 py-0">
              <span className="mr-1 inline-block h-1.5 w-1.5 rounded-full bg-green-400 animate-pulse" />
              {t('consciousness_master.live')}
            </Badge>
          )}
        </div>
      }
    >
      <div className="space-y-4 sm:space-y-6 p-4 sm:p-6">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs sm:text-sm text-muted-foreground">{t('consciousness_master.health_score')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-3">
                <div className="relative h-16 w-16">
                  <svg className="h-16 w-16 -rotate-90" viewBox="0 0 36 36">
                    <circle cx="18" cy="18" r="15.5" fill="none" stroke="hsl(var(--muted))" strokeWidth="3" />
                    <circle
                      cx="18" cy="18" r="15.5" fill="none"
                      stroke={healthColor}
                      strokeWidth="3"
                      strokeDasharray={`${(healthScore / 100) * 97.39} 97.39`}
                      strokeLinecap="round"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center text-sm font-bold">
                    {Math.round(healthScore)}
                  </div>
                </div>
                <div>
                  <div className="text-xl sm:text-2xl font-bold">{Math.round(healthScore)}<span className="text-xs text-muted-foreground">/100</span></div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs sm:text-sm text-muted-foreground">{t('consciousness_master.total_episodes')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xl sm:text-2xl font-bold">{totalEpisodes}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs sm:text-sm text-muted-foreground">{t('consciousness_master.avg_growth')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                <span className={`text-xl sm:text-2xl font-bold ${avgGrowth >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {avgGrowth >= 0 ? '↑' : '↓'} {(avgGrowth * 100).toFixed(1)}%
                </span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs sm:text-sm text-muted-foreground">{t('consciousness_master.avg_rating')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xl sm:text-2xl font-bold">
                {avgRating > 0 ? (
                  <span className="flex items-center gap-1">
                    {avgRating.toFixed(1)}
                    <span className="text-yellow-400 text-sm">{'★'.repeat(Math.round(avgRating))}</span>
                  </span>
                ) : '—'}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_master.qualia_title')}</CardTitle>
              <CardDescription>{t('consciousness_master.qualia_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              {Object.keys(currentQualia).length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <RadarChart data={qualiaRadarData}>
                    <PolarGrid stroke="hsl(var(--border))" />
                    <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} />
                    <PolarRadiusAxis angle={30} domain={[-1, 1]} tick={{ fontSize: 9 }} />
                    <Radar
                      name="Current"
                      dataKey="value"
                      stroke="#8b5cf6"
                      fill="#8b5cf6"
                      fillOpacity={0.25}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-48 items-center justify-center text-sm text-muted-foreground">
                  {t('consciousness_master.no_data')}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_master.beliefs_title')}</CardTitle>
              <CardDescription>{t('consciousness_master.beliefs_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              {beliefsBarData.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={beliefsBarData} layout="vertical" margin={{ left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                    <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" width={80} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'hsl(var(--card))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '8px',
                        fontSize: '12px',
                      }}
                    />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {beliefsBarData.map((entry, idx) => (
                        <rect key={idx} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-48 items-center justify-center text-sm text-muted-foreground">
                  {t('consciousness_master.no_data')}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_master.rating_title')}</CardTitle>
              <CardDescription>{t('consciousness_master.rating_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={ratingDist}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="rating" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <YAxis tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_master.growth_title')}</CardTitle>
              <CardDescription>{t('consciousness_master.growth_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              {growthData.length > 1 ? (
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={growthData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="time" tick={{ fontSize: 9 }} stroke="hsl(var(--muted-foreground))" />
                    <YAxis tick={{ fontSize: 9 }} stroke="hsl(var(--muted-foreground))" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'hsl(var(--card))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '8px',
                        fontSize: '12px',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Line type="monotone" dataKey="growth" stroke="#6366f1" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" dataKey="ma" stroke="#22c55e" strokeWidth={2} dot={false} strokeDasharray="5 5" />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-48 items-center justify-center text-sm text-muted-foreground">
                  {t('consciousness_master.no_data')}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_master.health_title')}</CardTitle>
              <CardDescription>{t('consciousness_master.health_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-center">
                  <div className="relative h-24 w-24">
                    <svg className="h-24 w-24 -rotate-90" viewBox="0 0 36 36">
                      <circle cx="18" cy="18" r="15.5" fill="none" stroke="hsl(var(--muted))" strokeWidth="3" />
                      <circle
                        cx="18" cy="18" r="15.5" fill="none"
                        stroke={healthColor}
                        strokeWidth="3"
                        strokeDasharray={`${(healthScore / 100) * 97.39} 97.39`}
                        strokeLinecap="round"
                      />
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                      <span className="text-lg font-bold">{Math.round(healthScore)}</span>
                      <span className="text-[9px] text-muted-foreground">/100</span>
                    </div>
                  </div>
                </div>
                <div className="space-y-1.5">
                  {[
                    { label: t('consciousness_health.episodes'), value: status?.episode_count ?? totalEpisodes },
                    { label: t('consciousness_health.avg_growth'), value: `${(avgGrowth * 100).toFixed(1)}%` },
                    { label: t('consciousness_health.positive_ratio'), value: `${((episodes.filter((e: any) => (e.growth_delta ?? 0) > 0).length / Math.max(1, episodes.length)) * 100).toFixed(0)}%` },
                    { label: t('consciousness_health.enabled'), value: status?.enabled ?? true ? t('consciousness_health.yes') : t('consciousness_health.no') },
                  ].map(item => (
                    <div key={item.label} className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">{item.label}</span>
                      <span className="font-medium">{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_master.diagnostics_title')}</CardTitle>
              <CardDescription>{t('consciousness_master.diagnostics_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              {diagnostics.length > 0 ? (
                <div className="space-y-2 max-h-[250px] overflow-y-auto">
                  {diagnostics.map((d: string, i: number) => (
                    <div key={i} className="flex items-start gap-2 text-sm">
                      <span className="mt-1 h-1.5 w-1.5 rounded-full bg-muted-foreground shrink-0" />
                      <span className="text-muted-foreground">{d}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex h-24 items-center justify-center text-sm text-muted-foreground">
                  {t('consciousness_master.no_diagnostics')}
                </div>
              )}
              {lastReflection && (
                <div className="mt-3 rounded-md bg-muted p-2 text-xs italic text-muted-foreground">
                  "{lastReflection}"
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_master.actions_title')}</CardTitle>
              <CardDescription>{t('consciousness_master.actions_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-2">
                <Button size="sm" variant="outline" onClick={handleSeed} disabled={seeding}>
                  {seeding ? t('consciousness_master.seeding') : t('consciousness_master.seed')}
                </Button>
                <Button size="sm" variant="outline" onClick={handleReflect} disabled={reflecting}>
                  {reflecting ? t('consciousness_master.reflecting') : t('consciousness_master.reflect')}
                </Button>
                <Button size="sm" variant="outline" onClick={handleExport} disabled={exporting}>
                  {exporting ? t('consciousness_master.exporting') : t('consciousness_master.export')}
                </Button>
                <Button size="sm" variant="outline" onClick={() => window.location.href = '/chat'}>
                  {t('consciousness_master.open_chat')}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {t('consciousness_master.recent_title')}
              <Badge variant="outline" className="text-xs">{Math.min(5, episodes.length)} shown</Badge>
            </CardTitle>
            <CardDescription>{t('consciousness_master.recent_desc')}</CardDescription>
          </CardHeader>
          <CardContent>
            {episodes.length > 0 ? (
              <div className="space-y-2">
                {[...episodes].reverse().slice(0, 5).map((ep: any, i: number) => (
                  <div key={i} className="flex gap-3 rounded-lg border p-2 sm:p-3 text-xs sm:text-sm">
                    <div className="flex flex-col items-center gap-1 text-[10px] sm:text-xs text-muted-foreground min-w-[48px]">
                      <span>{formatTimeAgo(ep.timestamp)}</span>
                      <span className={`font-mono ${(ep.growth_delta ?? 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {(ep.growth_delta ?? 0) >= 0 ? '+' : ''}{((ep.growth_delta ?? 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-[10px] sm:text-xs text-muted-foreground truncate">{ep.input_text}</div>
                      <div className="mt-1 text-xs sm:text-sm">{ep.self_insight}</div>
                      <div className="mt-1 flex flex-wrap gap-2">
                        {Object.entries(ep.qualia ?? {}).slice(0, 4).map(([k, v]) => (
                          <span key={k} className="text-[10px] text-muted-foreground">
                            {k}: {(v as number).toFixed(2)}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex h-24 items-center justify-center text-sm text-muted-foreground">
                {t('consciousness_master.no_episodes')}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
