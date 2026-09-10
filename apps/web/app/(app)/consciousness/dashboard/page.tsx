'use client'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { PUBLIC_API_URL } from '@/lib/config'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Skeleton, Switch,
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogPortal, DialogOverlay,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useConsciousnessLive } from '@/hooks/useConsciousnessLive'
import { useLocale } from '@/hooks/useLocale'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, AreaChart, Area,
} from 'recharts'

interface Episode {
  timestamp: number
  input_text: string
  self_insight: string
  growth_delta: number
  qualia: Record<string, number>
}

interface QualiaPoint {
  timestamp: number
  valence: number
  arousal: number
  novelty: number
  coherence: number
}

interface BeliefsPoint {
  timestamp: number
  step: number
  competence: number
  helpfulness: number
  creativity: number
  accuracy: number
  empathy: number
}

interface EvalReport {
  overall_score: number
  diagnostics: string[]
  [key: string]: unknown
}

const BELIEF_COLORS: Record<string, string> = {
  competence: '#6366f1',
  helpfulness: '#22c55e',
  creativity: '#f59e0b',
  accuracy: '#3b82f6',
  empathy: '#ec4899',
}

const QUALIA_COLORS: Record<string, string> = {
  valence: '#8b5cf6',
  arousal: '#ef4444',
  novelty: '#f59e0b',
  coherence: '#22c55e',
}

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

export default function ConsciousnessDashboardPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()
  const { isLive, lastUpdate, toggleLive, latestEvent } = useConsciousnessLive()
  const [loading, setLoading] = useState(true)
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [qualiaHistory, setQualiaHistory] = useState<QualiaPoint[]>([])
  const [beliefsHistory, setBeliefsHistory] = useState<BeliefsPoint[]>([])
  const [evalReport, setEvalReport] = useState<EvalReport | null>(null)
  const [totalEpisodes, setTotalEpisodes] = useState(0)
  const [currentBeliefs, setCurrentBeliefs] = useState<Record<string, number>>({})
  const [currentQualia, setCurrentQualia] = useState<Record<string, number>>({})
  const [seeding, setSeeding] = useState(false)
  const [reflecting, setReflecting] = useState(false)
  const [ratingEpisode, setRatingEpisode] = useState<number | null>(null)
  const [selectedEpisode, setSelectedEpisode] = useState<Episode | null>(null)
  const [consciousnessLevel, setConsciousnessLevel] = useState(1)
  const [exporting, setExporting] = useState(false)
  const [importing, setImporting] = useState(false)

  const fetchAll = useCallback(async () => {
    try {
      const [epRes, qRes, bRes, eRes, sRes, qNowRes, statusRes] = await Promise.allSettled([
        fetch(`${PUBLIC_API_URL}/consciousness/history/episodes?limit=50`),
        fetch(`${PUBLIC_API_URL}/consciousness/history/qualia?limit=100`),
        fetch(`${PUBLIC_API_URL}/consciousness/history/beliefs`),
        fetch(`${PUBLIC_API_URL}/consciousness/evaluate`),
        fetch(`${PUBLIC_API_URL}/consciousness/self-model`),
        fetch(`${PUBLIC_API_URL}/consciousness/qualia`),
        fetch(`${PUBLIC_API_URL}/consciousness/status`),
      ])

      if (epRes.status === 'fulfilled' && epRes.value.ok) {
        const json = await epRes.value.json()
        setEpisodes(json.data?.episodes ?? [])
        setTotalEpisodes(json.data?.total ?? 0)
      }
      if (qRes.status === 'fulfilled' && qRes.value.ok) {
        const json = await qRes.value.json()
        setQualiaHistory(json.data?.history ?? [])
      }
      if (bRes.status === 'fulfilled' && bRes.value.ok) {
        const json = await bRes.value.json()
        setBeliefsHistory(json.data?.beliefs ?? [])
        const last = json.data?.beliefs?.[json.data.beliefs.length - 1]
        if (last) {
          const { timestamp, step, ...rest } = last
          setCurrentBeliefs(rest)
        }
      }
      if (eRes.status === 'fulfilled' && eRes.value.ok) {
        const json = await eRes.value.json()
        setEvalReport(json.data ?? null)
      }
      if (sRes.status === 'fulfilled' && sRes.value.ok) {
        const json = await sRes.value.json()
        if (json.data?.self_beliefs) setCurrentBeliefs(json.data.self_beliefs)
      }
      if (qNowRes.status === 'fulfilled' && qNowRes.value.ok) {
        const json = await qNowRes.value.json()
        if (json.data) setCurrentQualia(json.data)
      }
      if (statusRes.status === 'fulfilled' && statusRes.value.ok) {
        const json = await statusRes.value.json()
        if (json.data?.level !== undefined) setConsciousnessLevel(json.data.level)
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { fetchAll() }, [fetchAll])

  useEffect(() => {
    if (!isLive || !latestEvent) return
    if (latestEvent.qualia) setCurrentQualia(latestEvent.qualia)
    if (latestEvent.beliefs) {
      setCurrentBeliefs(latestEvent.beliefs)
      setBeliefsHistory((prev) => {
        const point: BeliefsPoint = {
          timestamp: Date.now() / 1000,
          step: (prev[prev.length - 1]?.step ?? 0) + 1,
          competence: (latestEvent.beliefs as any).competence ?? 0,
          helpfulness: (latestEvent.beliefs as any).helpfulness ?? 0,
          creativity: (latestEvent.beliefs as any).creativity ?? 0,
          accuracy: (latestEvent.beliefs as any).accuracy ?? 0,
          empathy: (latestEvent.beliefs as any).empathy ?? 0,
        }
        return [...prev, point]
      })
    }
    if (latestEvent.qualia) {
      setQualiaHistory((prev) => {
        const point: QualiaPoint = {
          timestamp: Date.now() / 1000,
          valence: latestEvent.qualia.valence ?? 0,
          arousal: latestEvent.qualia.arousal ?? 0,
          novelty: latestEvent.qualia.novelty ?? 0,
          coherence: latestEvent.qualia.coherence ?? 0,
        }
        return [...prev, point]
      })
    }
    if (latestEvent.level !== undefined) setConsciousnessLevel(latestEvent.level)
    addToast('Consciousness state updated', 'success')
  }, [latestEvent, isLive, addToast])

  const handleSeed = async () => {
    setSeeding(true)
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/seed?count=30`, { method: 'POST' })
      if (res.ok) {
        addToast('30 episodes added', 'success')
        await fetchAll()
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setSeeding(false)
    }
  }

  const handleReflect = async () => {
    setReflecting(true)
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/reflect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: 'Dashboard reflection trigger' }),
      })
      if (res.ok) {
        addToast('New episode recorded', 'success')
        await fetchAll()
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setReflecting(false)
    }
  }

  const handleFeedback = async (episodeIndex: number, rating: number) => {
    setRatingEpisode(episodeIndex)
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ episode_index: episodeIndex, rating }),
      })
      if (res.ok) {
        addToast(`Rated ${rating}/5`, 'success')
        await fetchAll()
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setRatingEpisode(null)
    }
  }

  const handleLevelChange = async (newLevel: number) => {
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level: newLevel }),
      })
      if (res.ok) {
        setConsciousnessLevel(newLevel)
        addToast(`Level set to ${newLevel}`, 'success')
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }

  const handleExport = async () => {
    setExporting(true)
    try {
      const data = {
        episodes,
        qualiaHistory,
        beliefsHistory,
        currentBeliefs,
        currentQualia,
        evalReport,
        exportedAt: new Date().toISOString(),
      }
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `consciousness-export-${Date.now()}.json`
      a.click()
      URL.revokeObjectURL(url)
      addToast('Exported consciousness data', 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setExporting(false)
    }
  }

  const handleImport = async () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return
      setImporting(true)
      try {
        const text = await file.text()
        const data = JSON.parse(text)
        addToast(`Imported ${data.episodes?.length ?? 0} episodes`, 'success')
        await fetchAll()
      } catch (err) {
        addToast('Invalid JSON file', 'error')
      } finally {
        setImporting(false)
      }
    }
    input.click()
  }

  if (loading) {
    return (
      <PageContainer title="Consciousness Dashboard">
        <div className="space-y-6 p-6">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
          </div>
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
          <Skeleton className="h-48" />
        </div>
      </PageContainer>
    )
  }

  // Prepare qualia chart data with time labels
  const qualiaData = qualiaHistory.map((q) => ({
    ...q,
    time: formatTime(q.timestamp),
  }))

  // Prepare beliefs chart data with time labels
  const beliefsData = beliefsHistory.map((b) => ({
    ...b,
    time: formatTime(b.timestamp),
  }))

  // Growth data from episodes
  const growthData = episodes.map((ep) => ({
    time: formatTime(ep.timestamp),
    growth: ep.growth_delta,
    cumulative: episodes.slice(0, episodes.indexOf(ep) + 1)
      .reduce((sum, e) => sum + e.growth_delta, 0),
  }))

  return (
    <PageContainer title="Consciousness Dashboard">
      <div className="space-y-6 p-6">
        {/* Summary cards */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-5">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground">Total Episodes</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{totalEpisodes}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground">Quality Score</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {evalReport ? `${evalReport.overall_score.toFixed(0)}/100` : '—'}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground">Avg Growth</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {episodes.length > 0
                  ? `${(episodes.reduce((s, e) => s + e.growth_delta, 0) / episodes.length * 100).toFixed(1)}%`
                  : '—'}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground flex items-center gap-2">
                {t('consciousness_dashboard.autoRefresh')}
                {isLive && (
                  <Badge variant="default" className="text-[10px] px-1.5 py-0">
                    <span className="mr-1 inline-block h-1.5 w-1.5 rounded-full bg-green-400 animate-pulse" />
                    {t('consciousness_dashboard.live')}
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-3">
                <Switch checked={isLive} onCheckedChange={toggleLive} />
                <span className="text-xs text-muted-foreground">
                  {lastUpdate ? `Updated ${formatTimeAgo(lastUpdate / 1000)}` : 'No updates yet'}
                </span>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground">Actions</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              <Button size="sm" variant="outline" onClick={handleSeed} disabled={seeding}>
                {seeding ? 'Seeding...' : 'Seed 30'}
              </Button>
              <Button size="sm" variant="outline" onClick={handleReflect} disabled={reflecting}>
                {reflecting ? 'Reflecting...' : 'Reflect'}
              </Button>
              <Button size="sm" variant="outline" onClick={handleExport} disabled={exporting}>
                {exporting ? 'Exporting...' : 'Export'}
              </Button>
              <Button size="sm" variant="outline" onClick={handleImport} disabled={importing}>
                {importing ? 'Importing...' : 'Import'}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Response Quality & Last Reflection */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Response Quality</CardTitle>
              <CardDescription>How well responses match personality goals</CardDescription>
            </CardHeader>
            <CardContent>
              {evalReport ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Overall Score</span>
                    <span className="text-2xl font-bold">{evalReport.overall_score.toFixed(0)}/100</span>
                  </div>
                  <div className="h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${evalReport.overall_score}%`,
                        backgroundColor: evalReport.overall_score > 70 ? '#22c55e' : evalReport.overall_score > 40 ? '#f59e0b' : '#ef4444',
                      }}
                    />
                  </div>
                  {evalReport.diagnostics?.slice(0, 3).map((d: string, i: number) => (
                    <div key={i} className="text-xs text-muted-foreground">{d}</div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">No evaluation data yet</div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Last Reflection</CardTitle>
              <CardDescription>What the system is thinking right now</CardDescription>
            </CardHeader>
            <CardContent>
              {(() => {
                const status = evalReport as any
                const lastReflection = status?.last_reflection || episodes[episodes.length - 1]?.self_insight
                return lastReflection ? (
                  <div className="text-sm italic text-muted-foreground">"{lastReflection}"</div>
                ) : (
                  <div className="text-sm text-muted-foreground">No reflections yet. Click "Reflect" to generate one.</div>
                )
              })()}
            </CardContent>
          </Card>
        </div>

        {/* Consciousness level control */}
        <Card>
          <CardHeader>
            <CardTitle>Consciousness Level</CardTitle>
            <CardDescription>Adjust how deeply the system reflects on each interaction</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              {[0, 1, 2, 3].map((lvl) => (
                <Button
                  key={lvl}
                  size="sm"
                  variant={consciousnessLevel === lvl ? 'default' : 'outline'}
                  onClick={() => handleLevelChange(lvl)}
                  className="w-20"
                >
                  {lvl === 0 ? 'Off' : lvl === 1 ? 'Basic' : lvl === 2 ? 'Full' : 'Deep'}
                </Button>
              ))}
              <Badge variant="outline" className="ml-2">
                Current: {consciousnessLevel === 0 ? 'Off' : consciousnessLevel === 1 ? 'Basic' : consciousnessLevel === 2 ? 'Full' : 'Deep'}
              </Badge>
            </div>
          </CardContent>
        </Card>

        {/* Current state snapshot */}
        {(Object.keys(currentBeliefs).length > 0 || Object.keys(currentQualia).length > 0) && (
          <Card>
            <CardHeader>
              <CardTitle>Current State</CardTitle>
              <CardDescription>Live beliefs and qualia snapshot</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-6">
                {Object.keys(currentBeliefs).length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-muted-foreground mb-2">Beliefs</div>
                    <div className="space-y-1">
                      {Object.entries(currentBeliefs).map(([k, v]) => (
                        <div key={k} className="flex items-center gap-2 text-sm">
                          <span className="capitalize w-24">{k}</span>
                          <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-500"
                              style={{
                                width: `${Math.max(0, Math.min(100, (v as number) * 100))}%`,
                                backgroundColor: BELIEF_COLORS[k] || '#6366f1',
                              }}
                            />
                          </div>
                          <span className="text-xs text-muted-foreground w-10 text-right">
                            {((v as number) * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {Object.keys(currentQualia).length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-muted-foreground mb-2">Qualia</div>
                    <div className="space-y-1">
                      {Object.entries(currentQualia).map(([k, v]) => (
                        <div key={k} className="flex items-center gap-2 text-sm">
                          <span className="capitalize w-24">{k}</span>
                          <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-500"
                              style={{
                                width: `${Math.max(0, Math.min(100, ((v as number) + 1) / 2 * 100))}%`,
                                backgroundColor: QUALIA_COLORS[k] || '#8b5cf6',
                              }}
                            />
                          </div>
                          <span className="text-xs text-muted-foreground w-10 text-right">
                            {(v as number).toFixed(2)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Beliefs evolution chart */}
        <Card>
          <CardHeader>
            <CardTitle>Beliefs Evolution</CardTitle>
            <CardDescription>How self-beliefs change over interactions</CardDescription>
          </CardHeader>
          <CardContent>
            {beliefsData.length > 1 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={beliefsData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  {Object.entries(BELIEF_COLORS).map(([key, color]) => (
                    <Line
                      key={key}
                      type="monotone"
                      dataKey={key}
                      stroke={color}
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 4 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-sm text-muted-foreground">
                Need at least 2 data points. Chat with consciousness enabled to generate data.
              </div>
            )}
          </CardContent>
        </Card>

        {/* Qualia history chart */}
        <Card>
          <CardHeader>
            <CardTitle>Qualia History</CardTitle>
            <CardDescription>Emotional and cognitive state over time</CardDescription>
          </CardHeader>
          <CardContent>
            {qualiaData.length > 1 ? (
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={qualiaData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <YAxis domain={[-1, 1]} tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  {Object.entries(QUALIA_COLORS).map(([key, color]) => (
                    <Area
                      key={key}
                      type="monotone"
                      dataKey={key}
                      stroke={color}
                      fill={color}
                      fillOpacity={0.1}
                      strokeWidth={2}
                    />
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-sm text-muted-foreground">
                Need at least 2 data points. Chat with consciousness enabled to generate data.
              </div>
            )}
          </CardContent>
        </Card>

        {/* Growth trajectory */}
        {growthData.length > 1 && (
          <Card>
            <CardHeader>
              <CardTitle>Growth Trajectory</CardTitle>
              <CardDescription>Cumulative growth across interactions</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={growthData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <YAxis tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Area type="monotone" dataKey="cumulative" stroke="#6366f1" fill="#6366f1" fillOpacity={0.15} strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* Episode timeline */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Recent Episodes
              <Badge variant="outline" className="text-xs">{episodes.length} shown</Badge>
            </CardTitle>
            <CardDescription>Rate episodes to influence consciousness evolution</CardDescription>
          </CardHeader>
          <CardContent>
            {episodes.length > 0 ? (
              <div className="space-y-3 max-h-[500px] overflow-y-auto">
                {[...episodes].reverse().map((ep, i) => {
                  const realIndex = episodes.length - 1 - i
                  return (
                    <div key={i} className="flex gap-3 rounded-lg border p-3 text-sm cursor-pointer hover:bg-muted/50 transition-colors" onClick={() => setSelectedEpisode(ep)}>
                      <div className="flex flex-col items-center gap-1 text-xs text-muted-foreground min-w-[60px]">
                        <span>{formatTimeAgo(ep.timestamp)}</span>
                        <span className={`font-mono ${ep.growth_delta >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                          {ep.growth_delta >= 0 ? '+' : ''}{(ep.growth_delta * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-xs text-muted-foreground truncate">{ep.input_text}</div>
                        <div className="mt-1 text-sm">{ep.self_insight}</div>
                        <div className="mt-1 flex flex-wrap gap-2">
                          {Object.entries(ep.qualia).slice(0, 4).map(([k, v]) => (
                            <span key={k} className="text-[10px] text-muted-foreground">
                              {k}: {(v as number).toFixed(2)}
                            </span>
                          ))}
                        </div>
                        <div className="mt-2 flex gap-1">
                          {[1, 2, 3, 4, 5].map((r) => (
                            <Button
                              key={r}
                              size="sm"
                              variant={ratingEpisode === realIndex ? 'default' : 'outline'}
                              className="h-6 px-2 text-[10px]"
                              onClick={() => handleFeedback(realIndex, r)}
                              disabled={ratingEpisode === realIndex}
                            >
                              {r}
                            </Button>
                          ))}
                          <span className="text-[10px] text-muted-foreground ml-1 self-center">rate</span>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="flex h-[100px] items-center justify-center text-sm text-muted-foreground">
                No episodes yet. Chat with consciousness enabled to start recording.
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Episode Detail Modal */}
      <Dialog open={selectedEpisode !== null} onOpenChange={(open) => { if (!open) setSelectedEpisode(null) }}>
        <DialogPortal>
          <DialogOverlay />
          <DialogContent className="max-w-lg">
            <DialogHeader>
              <DialogTitle>Episode Detail</DialogTitle>
            </DialogHeader>
            {selectedEpisode && (
              <div className="space-y-4">
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-1">Input</div>
                  <div className="text-sm rounded-md bg-muted p-2">{selectedEpisode.input_text}</div>
                </div>
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-1">Self Insight</div>
                  <div className="text-sm">{selectedEpisode.self_insight}</div>
                </div>
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-1">Growth Delta</div>
                  <div className={`text-sm font-mono ${selectedEpisode.growth_delta >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {selectedEpisode.growth_delta >= 0 ? '+' : ''}{(selectedEpisode.growth_delta * 100).toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2">Qualia Breakdown</div>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(selectedEpisode.qualia).map(([k, v]) => (
                      <div key={k} className="flex items-center gap-2">
                        <span className="text-xs capitalize w-20">{k}</span>
                        <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${Math.max(0, Math.min(100, ((v as number) + 1) / 2 * 100))}%`,
                              backgroundColor: QUALIA_COLORS[k] || '#8b5cf6',
                            }}
                          />
                        </div>
                        <span className="text-[10px] text-muted-foreground w-8 text-right">
                          {(v as number).toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-1">Timestamp</div>
                  <div className="text-xs text-muted-foreground">
                    {new Date(selectedEpisode.timestamp * 1000).toLocaleString()}
                  </div>
                </div>
                <div className="flex gap-2 pt-2">
                  {[1, 2, 3, 4, 5].map((r) => (
                    <Button
                      key={r}
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        const idx = episodes.indexOf(selectedEpisode)
                        if (idx >= 0) handleFeedback(idx, r)
                        setSelectedEpisode(null)
                      }}
                    >
                      {r}
                    </Button>
                  ))}
                  <span className="text-xs text-muted-foreground ml-1 self-center">rate this episode</span>
                </div>
              </div>
            )}
          </DialogContent>
        </DialogPortal>
      </Dialog>
    </PageContainer>
  )
}
