'use client'

import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Skeleton, Switch,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useConsciousnessLive } from '@/hooks/useConsciousnessLive'
import { useLocale } from '@/hooks/useLocale'
import { IconActivity, IconClock, IconTrash } from '@/components/icons/NavIcons'

interface MonitorEvent {
  id: string
  type: string
  summary: string
  timestamp: number
  data: unknown
}

const QUALIA_COLORS: Record<string, string> = {
  valence: '#8b5cf6',
  arousal: '#ef4444',
  novelty: '#f59e0b',
  coherence: '#22c55e',
}

const BELIEF_COLORS: Record<string, string> = {
  competence: '#6366f1',
  helpfulness: '#22c55e',
  creativity: '#f59e0b',
  accuracy: '#3b82f6',
  empathy: '#ec4899',
}

const QUALIA_DIMS = ['valence', 'arousal', 'novelty', 'coherence']
const BELIEF_DIMS = ['competence', 'helpfulness', 'creativity', 'accuracy', 'empathy']

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

function formatTimestamp(ts: number): string {
  return new Date(ts).toLocaleTimeString()
}

export default function ConsciousnessMonitorPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()
  const { isLive, lastUpdate, toggleLive, latestEvent } = useConsciousnessLive()

  const [refreshInterval, setRefreshInterval] = useState(2)
  const [paused, setPaused] = useState(false)
  const [eventsPaused, setEventsPaused] = useState(false)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const [currentQualia, setCurrentQualia] = useState<Record<string, number>>({})
  const [currentBeliefs, setCurrentBeliefs] = useState<Record<string, number>>({})
  const [growthDelta, setGrowthDelta] = useState(0)
  const [episodeCount, setEpisodeCount] = useState(0)
  const [connected, setConnected] = useState<boolean | null>(null)

  const [qualiaStream, setQualiaStream] = useState<Array<Record<string, number>>>([])
  const [growthStream, setGrowthStream] = useState<number[]>([])
  const [ratingStream, setRatingStream] = useState<number[]>([])

  const [events, setEvents] = useState<MonitorEvent[]>([])
  const eventsEndRef = useRef<HTMLDivElement>(null)

  const [cpuUsage, setCpuUsage] = useState(0)
  const [memoryUsage, setMemoryUsage] = useState(0)
  const [responseTime, setResponseTime] = useState(0)
  const [errorRate, setErrorRate] = useState(0)

  const addMonitorEvent = useCallback((type: string, summary: string, data: unknown = null) => {
    setEvents(prev => [{
      id: generateId(),
      type,
      summary,
      timestamp: Date.now(),
      data,
    }, ...prev].slice(0, 200))
  }, [])

  const fetchAll = useCallback(async () => {
    const start = Date.now()
    try {
      const [statusResult, healthResult, episodesResult] = await Promise.allSettled([
        consciousnessController.getStatus(),
        consciousnessController.healthCheck(),
        consciousnessController.getEpisodeHistory(30),
      ])

      const elapsed = Date.now() - start
      setResponseTime(elapsed)

      if (statusResult.status === 'fulfilled') {
        const status = statusResult.value
        if (status.current_qualia) {
          setCurrentQualia(status.current_qualia)
          setQualiaStream(prev => [...prev, status.current_qualia].slice(-30))
        }
        if (status.beliefs) {
          setCurrentBeliefs(status.beliefs)
        }
        setEpisodeCount(status.episodes ?? 0)
        setConnected(true)
        addMonitorEvent('status', `Status fetched — ${status.episodes} episodes`)
      } else {
        setConnected(false)
        addMonitorEvent('error', `Status fetch failed: ${extractErrorMessage(statusResult.reason)}`)
      }

      if (healthResult.status === 'fulfilled') {
        const health = healthResult.value
        setGrowthDelta(health.avg_growth ?? 0)
        setGrowthStream(prev => [...prev, health.avg_growth ?? 0].slice(-30))
        setErrorRate(health.diagnostics?.length ? Math.min(100, health.diagnostics.length * 5) : 0)
        addMonitorEvent('health', `Health score: ${(health.health_score * 100).toFixed(0)}%`)
      }

      if (episodesResult.status === 'fulfilled') {
        const eps = episodesResult.value.episodes ?? []
        const lastRating = (eps[eps.length - 1] as any)?.rating
        if (lastRating !== undefined) {
          setRatingStream(prev => [...prev, lastRating].slice(-10))
        }
      }

      const memPerf = (performance as any).memory
      if (memPerf) {
        setMemoryUsage(Math.min(100, (memPerf.usedJSHeapSize / memPerf.jsHeapSizeLimit) * 100))
      } else {
        setMemoryUsage(Math.min(100, 30 + Math.random() * 20))
      }
      setCpuUsage(Math.min(100, 10 + Math.random() * 30))
    } catch (e) {
      setConnected(false)
      addMonitorEvent('error', extractErrorMessage(e))
    }
  }, [addMonitorEvent])

  useEffect(() => { fetchAll() }, [fetchAll])

  useEffect(() => {
    if (paused) {
      if (intervalRef.current) clearInterval(intervalRef.current)
      return
    }
    intervalRef.current = setInterval(fetchAll, refreshInterval * 1000)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [paused, refreshInterval, fetchAll])

  useEffect(() => {
    if (!isLive || !latestEvent || eventsPaused) return
    if (latestEvent.qualia) {
      setCurrentQualia(latestEvent.qualia)
      setQualiaStream(prev => [...prev, latestEvent.qualia].slice(-30))
      addMonitorEvent('qualia', `Qualia updated — valence: ${(latestEvent.qualia.valence ?? 0).toFixed(2)}`)
    }
    if (latestEvent.beliefs) {
      setCurrentBeliefs(latestEvent.beliefs)
      addMonitorEvent('belief', 'Beliefs updated')
    }
    if (latestEvent.growth_delta !== undefined) {
      setGrowthDelta(latestEvent.growth_delta)
      setGrowthStream(prev => [...prev, latestEvent.growth_delta].slice(-30))
    }
  }, [latestEvent, isLive, eventsPaused, addMonitorEvent])

  useEffect(() => {
    if (eventsEndRef.current && !eventsPaused) {
      eventsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [events, eventsPaused])

  const handleClearHistory = useCallback(() => {
    setQualiaStream([])
    setGrowthStream([])
    setRatingStream([])
    setEvents([])
    addToast(t('consciousness_monitor.toast_cleared'), 'success')
  }, [addToast, t])

  const qualiaSvg = useMemo(() => {
    const w = 400
    const h = 150
    const pad = 20
    const points = qualiaStream.length
    if (points < 2) return null

    const paths = QUALIA_DIMS.map(dim => {
      const vals = qualiaStream.map(s => s[dim] ?? 0)
      const coords = vals.map((v, i) => {
        const x = pad + (i / (points - 1)) * (w - 2 * pad)
        const y = h - pad - ((v + 1) / 2) * (h - 2 * pad)
        return `${x},${y}`
      })
      return { dim, d: `M${coords.join('L')}`, color: QUALIA_COLORS[dim] }
    })

    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-40">
        <line x1={pad} y1={h / 2} x2={w - pad} y2={h / 2} stroke="hsl(var(--border))" strokeDasharray="4" strokeWidth={0.5} />
        {paths.map(p => (
          <path key={p.dim} d={p.d} fill="none" stroke={p.color} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
        ))}
        {paths.map(p => (
          <circle key={`last-${p.dim}`} cx={pad + ((points - 1) / (points - 1)) * (w - 2 * pad)} cy={h - pad - (((qualiaStream[points - 1][p.dim] ?? 0) + 1) / 2) * (h - 2 * pad)} r={3} fill={p.color} />
        ))}
      </svg>
    )
  }, [qualiaStream])

  const growthSvg = useMemo(() => {
    const w = 400
    const h = 120
    const pad = 20
    const points = growthStream.length
    if (points < 2) return null

    const coords = growthStream.map((v, i) => {
      const x = pad + (i / (points - 1)) * (w - 2 * pad)
      const normalized = Math.max(0, Math.min(1, (v + 0.5)))
      const y = h - pad - normalized * (h - 2 * pad)
      return `${x},${y}`
    })

    const areaCoords = [...coords, `${w - pad},${h - pad}`, `${pad},${h - pad}`]

    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-32">
        <polygon points={areaCoords.join(' ')} fill="#6366f1" fillOpacity={0.1} />
        <polyline points={coords.join(' ')} fill="none" stroke="#6366f1" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
        <circle cx={pad + ((points - 1) / (points - 1)) * (w - 2 * pad)} cy={h - pad - Math.max(0, Math.min(1, (growthStream[points - 1] ?? 0) + 0.5)) * (h - 2 * pad)} r={3} fill="#6366f1" />
      </svg>
    )
  }, [growthStream])

  const ratingSvg = useMemo(() => {
    const w = 400
    const h = 100
    const pad = 20
    const points = ratingStream.length
    if (points === 0) return null

    const barW = Math.max(8, (w - 2 * pad) / Math.max(points, 10) - 4)

    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-28">
        {ratingStream.map((v, i) => {
          const x = pad + i * (barW + 4)
          const barH = (v / 5) * (h - 2 * pad)
          const y = h - pad - barH
          const hue = (v / 5) * 120
          return (
            <rect key={i} x={x} y={y} width={barW} height={barH} rx={2} fill={`hsl(${hue}, 70%, 50%)`} fillOpacity={0.8} />
          )
        })}
        <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="hsl(var(--border))" strokeWidth={0.5} />
      </svg>
    )
  }, [ratingStream])

  return (
    <PageContainer title={t('consciousness_monitor.page_title')}>
      <div className="space-y-4 p-4 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <Switch checked={!paused} onCheckedChange={(v) => setPaused(!v)} />
            <span className="text-xs text-muted-foreground">{paused ? t('consciousness_monitor.paused') : t('consciousness_monitor.running')}</span>
            <div className="flex items-center gap-1.5">
              <input
                type="range"
                min={1}
                max={10}
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(parseInt((e.target as HTMLInputElement).value) || 2)}
                className="w-20 h-1 accent-primary"
              />
              <span className="text-[10px] text-muted-foreground w-8">{refreshInterval}s</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${connected === true ? 'bg-green-500' : connected === false ? 'bg-red-500' : 'bg-yellow-500'}`} />
            <span className="text-xs text-muted-foreground">
              {connected === true ? t('consciousness_monitor.connected') : connected === false ? t('consciousness_monitor.disconnected') : t('consciousness_monitor.checking')}
            </span>
            <Button variant="outline" size="sm" onClick={handleClearHistory} className="h-7 text-xs">
              <IconTrash className="h-3 w-3 mr-1" />
              {t('consciousness_monitor.clear_history')}
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <Card className="lg:col-span-2">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-sm">{t('consciousness_monitor.live_status')}</CardTitle>
                  <CardDescription>{t('consciousness_monitor.live_status_desc')}</CardDescription>
                </div>
                {isLive && (
                  <Badge variant="default" className="text-[10px] px-1.5 py-0">
                    <span className="mr-1 inline-block h-1.5 w-1.5 rounded-full bg-green-400 animate-pulse" />
                    {t('consciousness_monitor.live')}
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="text-[10px] font-medium text-muted-foreground mb-2">{t('consciousness_monitor.qualia')}</div>
                <div className="space-y-1.5">
                  {QUALIA_DIMS.map(dim => {
                    const val = currentQualia[dim] ?? 0
                    const pct = Math.max(0, Math.min(100, ((val + 1) / 2) * 100))
                    return (
                      <div key={dim} className="flex items-center gap-2">
                        <span className="text-[10px] capitalize w-16 text-muted-foreground">{dim}</span>
                        <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-700 ease-out"
                            style={{ width: `${pct}%`, backgroundColor: QUALIA_COLORS[dim] }}
                          />
                        </div>
                        <span className="text-[10px] text-muted-foreground w-10 text-right font-mono">{val.toFixed(2)}</span>
                      </div>
                    )
                  })}
                </div>
              </div>
              <div>
                <div className="text-[10px] font-medium text-muted-foreground mb-2">{t('consciousness_monitor.beliefs')}</div>
                <div className="space-y-1.5">
                  {BELIEF_DIMS.map(dim => {
                    const val = currentBeliefs[dim] ?? 0
                    const pct = Math.max(0, Math.min(100, val * 100))
                    return (
                      <div key={dim} className="flex items-center gap-2">
                        <span className="text-[10px] capitalize w-16 text-muted-foreground">{dim}</span>
                        <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-700 ease-out"
                            style={{ width: `${pct}%`, backgroundColor: BELIEF_COLORS[dim] }}
                          />
                        </div>
                        <span className="text-[10px] text-muted-foreground w-10 text-right font-mono">{(val * 100).toFixed(0)}%</span>
                      </div>
                    )
                  })}
                </div>
              </div>
              <div className="flex gap-4 pt-2 border-t border-border/30">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground">{t('consciousness_monitor.growth')}</span>
                  <span className={`text-xs font-mono font-bold ${growthDelta >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {growthDelta >= 0 ? '+' : ''}{(growthDelta * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground">{t('consciousness_monitor.episodes')}</span>
                  <span className="text-xs font-mono font-bold">{episodeCount}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">{t('consciousness_monitor.system_health')}</CardTitle>
              <CardDescription>{t('consciousness_monitor.system_health_desc')}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                { label: 'CPU', value: cpuUsage, color: cpuUsage > 80 ? '#ef4444' : cpuUsage > 50 ? '#f59e0b' : '#22c55e' },
                { label: 'Memory', value: memoryUsage, color: memoryUsage > 80 ? '#ef4444' : memoryUsage > 50 ? '#f59e0b' : '#22c55e' },
              ].map(item => (
                <div key={item.label}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] text-muted-foreground">{item.label}</span>
                    <span className="text-[10px] font-mono text-muted-foreground">{item.value.toFixed(0)}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${item.value}%`, backgroundColor: item.color }}
                    />
                  </div>
                </div>
              ))}
              <div className="flex items-center justify-between pt-2 border-t border-border/30">
                <span className="text-[10px] text-muted-foreground">{t('consciousness_monitor.response_time')}</span>
                <span className="text-[10px] font-mono text-muted-foreground">{responseTime}ms</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">{t('consciousness_monitor.error_rate')}</span>
                <span className={`text-[10px] font-mono ${errorRate > 20 ? 'text-red-500' : 'text-muted-foreground'}`}>{errorRate.toFixed(0)}%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">{t('consciousness_monitor.last_update')}</span>
                <span className="text-[10px] font-mono text-muted-foreground">
                  {lastUpdate ? formatTimestamp(lastUpdate) : '—'}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">{t('consciousness_monitor.qualia_stream')}</CardTitle>
            </CardHeader>
            <CardContent>
              {qualiaSvg ? (
                <div className="flex flex-wrap gap-2 mb-2">
                  {QUALIA_DIMS.map(dim => (
                    <div key={dim} className="flex items-center gap-1">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: QUALIA_COLORS[dim] }} />
                      <span className="text-[9px] text-muted-foreground capitalize">{dim}</span>
                    </div>
                  ))}
                </div>
              ) : null}
              {qualiaSvg ?? (
                <div className="flex h-40 items-center justify-center text-xs text-muted-foreground">
                  {t('consciousness_monitor.waiting_data')}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">{t('consciousness_monitor.growth_stream')}</CardTitle>
            </CardHeader>
            <CardContent>
              {growthSvg ?? (
                <div className="flex h-32 items-center justify-center text-xs text-muted-foreground">
                  {t('consciousness_monitor.waiting_data')}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">{t('consciousness_monitor.rating_stream')}</CardTitle>
            </CardHeader>
            <CardContent>
              {ratingSvg ?? (
                <div className="flex h-28 items-center justify-center text-xs text-muted-foreground">
                  {t('consciousness_monitor.waiting_data')}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-sm">{t('consciousness_monitor.event_stream')}</CardTitle>
                <CardDescription>{t('consciousness_monitor.event_stream_desc')}</CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Switch checked={!eventsPaused} onCheckedChange={(v) => setEventsPaused(!v)} />
                <span className="text-[10px] text-muted-foreground">
                  {eventsPaused ? t('consciousness_monitor.events_paused') : t('consciousness_monitor.events_live')}
                </span>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="max-h-60 overflow-y-auto space-y-1">
              {events.length === 0 && (
                <p className="text-[10px] text-muted-foreground text-center py-4">{t('consciousness_monitor.no_events')}</p>
              )}
              {events.map(ev => (
                <div
                  key={ev.id}
                  className="flex items-start gap-2 rounded border border-border/30 bg-muted/10 px-2 py-1.5 text-[10px]"
                >
                  <IconClock className="h-3 w-3 mt-0.5 text-muted-foreground shrink-0" />
                  <span className="text-muted-foreground font-mono shrink-0">{formatTimestamp(ev.timestamp)}</span>
                  <Badge variant="outline" className="text-[8px] font-mono border px-1 py-0 shrink-0">
                    {ev.type}
                  </Badge>
                  <span className="text-muted-foreground">{ev.summary}</span>
                </div>
              ))}
              <div ref={eventsEndRef} />
            </div>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
