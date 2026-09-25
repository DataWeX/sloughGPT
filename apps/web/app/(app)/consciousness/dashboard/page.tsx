'use client'

import { useState, useEffect, useCallback } from 'react'
import { formatDateTime, formatTimeShort, toDateSeconds } from '@/lib/time-format'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Skeleton,
  Switch,
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogPortal,
  DialogOverlay,
  FoldSection,
  KpiGrid,
  SectionHeader,
  StatCard,
  StatusDot,
  cn,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useConsciousnessLive } from '@/hooks/useConsciousnessLive'
import { useLocale } from '@/hooks/useLocale'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
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

// Chart series routed through the theme tokens — chart-N follows the active
// theme/palette, so the dashboard never hardcodes a hue of its own.
const BELIEF_COLORS: Record<string, string> = {
  competence: 'var(--chart-1)',
  helpfulness: 'var(--chart-2)',
  creativity: 'var(--chart-3)',
  accuracy: 'var(--chart-4)',
  empathy: 'var(--chart-5)',
}

const QUALIA_COLORS: Record<string, string> = {
  valence: 'var(--chart-1)',
  arousal: 'var(--chart-5)',
  novelty: 'var(--chart-3)',
  coherence: 'var(--chart-2)',
}

function formatTime(ts: number): string {
  return formatTimeShort(toDateSeconds(ts))
}

function formatTimeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function levelName(lvl: number): string {
  return lvl === 0 ? 'Off' : lvl === 1 ? 'Basic' : lvl === 2 ? 'Full' : 'Deep'
}

export default function ConsciousnessDashboardPage() {
  const addToast = useToastStore((state) => state.addToast)
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
      const [epResult, qResult, bResult, eResult, sResult, qNowResult, statusResult] =
        await Promise.allSettled([
          consciousnessController.getEpisodeHistory(50),
          consciousnessController.getQualiaHistory(100),
          consciousnessController.getBeliefsHistory(),
          consciousnessController.evaluate(),
          consciousnessController.getSelfModel(),
          consciousnessController.getQualia(),
          consciousnessController.getStatus(),
        ])

      if (epResult.status === 'fulfilled') {
        setEpisodes((epResult.value.episodes ?? []) as unknown as Episode[])
        setTotalEpisodes(epResult.value.total ?? 0)
      }
      if (qResult.status === 'fulfilled') {
        setQualiaHistory((qResult.value.history ?? []) as unknown as QualiaPoint[])
      }
      if (bResult.status === 'fulfilled') {
        setBeliefsHistory((bResult.value.beliefs ?? []) as unknown as BeliefsPoint[])
        const last = bResult.value.beliefs?.[bResult.value.beliefs.length - 1]
        if (last) {
          const { timestamp, step, ...rest } = last
          setCurrentBeliefs(rest)
        }
      }
      if (eResult.status === 'fulfilled') {
        setEvalReport(eResult.value ?? null)
      }
      if (sResult.status === 'fulfilled') {
        if (sResult.value.self_beliefs) setCurrentBeliefs(sResult.value.self_beliefs)
      }
      if (qNowResult.status === 'fulfilled') {
        if (qNowResult.value) setCurrentQualia(qNowResult.value)
      }
      if (statusResult.status === 'fulfilled') {
        if (statusResult.value.level !== undefined) setConsciousnessLevel(statusResult.value.level)
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    fetchAll()
  }, [fetchAll])

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

  const handleFeedback = async (episodeIndex: number, rating: number) => {
    setRatingEpisode(episodeIndex)
    try {
      await consciousnessController.submitFeedback({ episode_index: episodeIndex, rating })
      addToast(`Rated ${rating}/5`, 'success')
      await fetchAll()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setRatingEpisode(null)
    }
  }

  const handleLevelChange = async (newLevel: number) => {
    try {
      await consciousnessController.updateConfig({ level: newLevel })
      setConsciousnessLevel(newLevel)
      addToast(`Level set to ${newLevel}`, 'success')
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
    return <PageContainer title="Consciousness Dashboard" loadingGrid />
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
    cumulative: episodes
      .slice(0, episodes.indexOf(ep) + 1)
      .reduce((sum, e) => sum + e.growth_delta, 0),
  }))

  const avgGrowth =
    episodes.length > 0
      ? `${((episodes.reduce((s, e) => s + e.growth_delta, 0) / episodes.length) * 100).toFixed(1)}%`
      : '—'

  const score = evalReport ? evalReport.overall_score : null

  return (
    <PageContainer
      title={
        <Card className="border-primary/20 bg-gradient-to-br from-primary/[0.04] via-transparent to-accent/[0.03]">
          <CardContent className="p-4 sm:p-5">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <StatusDot
                    tone="success"
                    pulse
                    showLabel
                    label={
                      isLive
                        ? t('consciousness_dashboard.live')
                        : t('consciousness_dashboard.autoRefresh')
                    }
                  />
                </div>
                <h1 className="sl-h1">Consciousness Dashboard</h1>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <Switch
                  checked={isLive}
                  onCheckedChange={toggleLive}
                  aria-label="Toggle live updates"
                />
              </div>
            </div>
            {lastUpdate && (
              <div className="flex items-center gap-4 mt-3 pt-3 border-t border-border/40">
                <span className="text-xs text-muted-foreground">
                  Updated {formatTimeAgo(lastUpdate / 1000)}
                </span>
              </div>
            )}
          </CardContent>
        </Card>
      }
      headerRight={
        <div className="flex flex-wrap gap-2">
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
        </div>
      }
    >
      {/* Focal metrics */}
      <KpiGrid columns={4}>
        <StatCard label="Total Episodes" value={totalEpisodes} numeric />
        <StatCard
          label="Quality Score"
          value={score !== null ? `${score.toFixed(0)}/100` : '—'}
          numeric={score !== null}
          description="How well responses match personality goals"
        />
        <StatCard label="Avg Growth" value={avgGrowth} numeric />
        <StatCard
          label="Consciousness Level"
          value={levelName(consciousnessLevel)}
          description={`Depth of reflection · ${isLive ? 'live' : 'auto-refresh'}`}
        />
      </KpiGrid>

      {/* Response quality & reflection */}
      <SectionHeader
        title="Response Quality & Reflection"
        description="How responses track personality goals, and the latest self-insight"
      />
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Response Quality</CardTitle>
            <CardDescription>How well responses match personality goals</CardDescription>
          </CardHeader>
          <CardContent>
            {score !== null ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Overall Score</span>
                  <span className="text-xl font-bold tracking-tight">{score.toFixed(0)}/100</span>
                </div>
                <div
                  className="h-2 rounded-full bg-muted overflow-hidden"
                  role="progressbar"
                  aria-valuenow={score}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <div
                    className={cn(
                      'h-full rounded-full transition-all duration-500',
                      score > 70 ? 'bg-success' : score > 40 ? 'bg-warning' : 'bg-destructive',
                    )}
                    style={{ width: `${score}%` }}
                  />
                </div>
                {evalReport?.diagnostics?.slice(0, 3).map((d: string, i: number) => (
                  <div key={i} className="text-xs text-muted-foreground">
                    {d}
                  </div>
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
              const lastReflection =
                status?.last_reflection || episodes[episodes.length - 1]?.self_insight
              return lastReflection ? (
                <div className="text-sm italic text-muted-foreground">"{lastReflection}"</div>
              ) : (
                <div className="text-sm text-muted-foreground">
                  No reflections yet. Click "Reflect" to generate one.
                </div>
              )
            })()}
          </CardContent>
        </Card>
      </div>

      {/* Evolution */}
      <SectionHeader
        title="Evolution"
        description="Beliefs, qualia, and growth over interactions"
      />
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
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10 }}
                  stroke="hsl(var(--muted-foreground))"
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontSize: 10 }}
                  stroke="hsl(var(--muted-foreground))"
                />
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
            <div className="flex h-48 sm:h-64 items-center justify-center text-sm text-muted-foreground">
              Need at least 2 data points. Chat with consciousness enabled to generate data.
            </div>
          )}
        </CardContent>
      </Card>

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
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10 }}
                  stroke="hsl(var(--muted-foreground))"
                />
                <YAxis
                  domain={[-1, 1]}
                  tick={{ fontSize: 10 }}
                  stroke="hsl(var(--muted-foreground))"
                />
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
            <div className="flex h-48 sm:h-64 items-center justify-center text-sm text-muted-foreground">
              Need at least 2 data points. Chat with consciousness enabled to generate data.
            </div>
          )}
        </CardContent>
      </Card>

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
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10 }}
                  stroke="hsl(var(--muted-foreground))"
                />
                <YAxis tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="cumulative"
                  stroke="var(--chart-1)"
                  fill="var(--chart-1)"
                  fillOpacity={0.15}
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Timeline */}
      <SectionHeader
        title="Recent Episodes"
        description="Rate episodes to influence consciousness evolution"
        action={
          <Badge variant="outline" className="text-xs">
            {episodes.length} shown
          </Badge>
        }
      />
      <Card>
        <CardContent>
          {episodes.length > 0 ? (
            <div className="space-y-3 max-h-[500px] overflow-y-auto">
              {[...episodes].reverse().map((ep, i) => {
                const realIndex = episodes.length - 1 - i
                return (
                  <div
                    key={i}
                    className="flex gap-2 sm:gap-3 rounded-lg border p-2 sm:p-3 text-xs sm:text-sm cursor-pointer hover:bg-muted/50 transition-colors"
                    onClick={() => setSelectedEpisode(ep)}
                  >
                    <div className="flex flex-col items-center gap-1 text-[10px] sm:text-xs text-muted-foreground min-w-[48px] sm:min-w-[60px]">
                      <span>{formatTimeAgo(ep.timestamp)}</span>
                      <span
                        className={`font-mono ${ep.growth_delta >= 0 ? 'text-success' : 'text-destructive'}`}
                      >
                        {ep.growth_delta >= 0 ? '+' : ''}
                        {(ep.growth_delta * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-[10px] sm:text-xs text-muted-foreground truncate">
                        {ep.input_text}
                      </div>
                      <div className="mt-1 text-xs sm:text-sm">{ep.self_insight}</div>
                      <div className="mt-1 flex flex-wrap gap-2">
                        {Object.entries(ep.qualia)
                          .slice(0, 4)
                          .map(([k, v]) => (
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
                        <span className="text-[10px] text-muted-foreground ml-1 self-center">
                          rate
                        </span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="flex h-24 sm:h-[100px] items-center justify-center text-sm text-muted-foreground">
              No episodes yet. Chat with consciousness enabled to start recording.
            </div>
          )}
        </CardContent>
      </Card>

      {/* Secondary controls — collapsible so the body stays focused */}
      <FoldSection heading="Consciousness Level" open>
        <p className="mb-3 text-sm text-muted-foreground">
          Adjust how deeply the system reflects on each interaction
        </p>
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {[0, 1, 2, 3].map((lvl) => (
            <Button
              key={lvl}
              size="sm"
              variant={consciousnessLevel === lvl ? 'default' : 'outline'}
              onClick={() => handleLevelChange(lvl)}
              className="h-8 sm:h-9 w-16 sm:w-20 text-xs sm:text-sm"
            >
              {lvl === 0 ? 'Off' : lvl === 1 ? 'Basic' : lvl === 2 ? 'Full' : 'Deep'}
            </Button>
          ))}
          <Badge variant="outline" className="ml-2">
            Current: {levelName(consciousnessLevel)}
          </Badge>
        </div>
      </FoldSection>

      {(Object.keys(currentBeliefs).length > 0 || Object.keys(currentQualia).length > 0) && (
        <FoldSection heading="Current State">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-6">
            {Object.keys(currentBeliefs).length > 0 && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2">Beliefs</div>
                <div className="space-y-1">
                  {Object.entries(currentBeliefs).map(([k, v]) => (
                    <div key={k} className="flex items-center gap-2 text-xs sm:text-sm">
                      <span className="capitalize w-20 sm:w-24">{k}</span>
                      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${Math.max(0, Math.min(100, (v as number) * 100))}%`,
                            backgroundColor: BELIEF_COLORS[k] || 'var(--chart-1)',
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
                    <div key={k} className="flex items-center gap-2 text-xs sm:text-sm">
                      <span className="capitalize w-20 sm:w-24">{k}</span>
                      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${Math.max(0, Math.min(100, (((v as number) + 1) / 2) * 100))}%`,
                            backgroundColor: QUALIA_COLORS[k] || 'var(--chart-1)',
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
        </FoldSection>
      )}

      {/* Episode Detail Modal */}
      <Dialog
        open={selectedEpisode !== null}
        onOpenChange={(open) => {
          if (!open) setSelectedEpisode(null)
        }}
      >
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
                  <div className="text-sm rounded-md bg-muted p-2">
                    {selectedEpisode.input_text}
                  </div>
                </div>
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-1">Self Insight</div>
                  <div className="text-sm">{selectedEpisode.self_insight}</div>
                </div>
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-1">Growth Delta</div>
                  <div
                    className={`text-sm font-mono ${selectedEpisode.growth_delta >= 0 ? 'text-success' : 'text-destructive'}`}
                  >
                    {selectedEpisode.growth_delta >= 0 ? '+' : ''}
                    {(selectedEpisode.growth_delta * 100).toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2">
                    Qualia Breakdown
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(selectedEpisode.qualia).map(([k, v]) => (
                      <div key={k} className="flex items-center gap-2">
                        <span className="text-xs capitalize w-20">{k}</span>
                        <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${Math.max(0, Math.min(100, (((v as number) + 1) / 2) * 100))}%`,
                              backgroundColor: QUALIA_COLORS[k] || 'var(--chart-1)',
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
                    {formatDateTime(toDateSeconds(selectedEpisode.timestamp))}
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
                  <span className="text-xs text-muted-foreground ml-1 self-center">
                    rate this episode
                  </span>
                </div>
              </div>
            )}
          </DialogContent>
        </DialogPortal>
      </Dialog>
    </PageContainer>
  )
}
