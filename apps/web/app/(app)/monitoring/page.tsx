'use client'
export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { systemController, type DetailedHealth, type SystemMetrics, type SystemInfo, type DiskUsage, type GPUInfo } from '@/lib/system-controller'
import { knowledgeController } from '@/lib/knowledge-controller'
import { benchmarkController } from '@/lib/benchmark-controller'
import { multimodalController } from '@/lib/controllers'
import dynamicNext from 'next/dynamic'

const SystemChart = dynamicNext(() => import('@/components/monitoring/SystemChart').then(m => m.SystemChart), {
  ssr: false,
  loading: () => <div className="h-40 w-full animate-pulse bg-muted rounded-lg" />,
})
import { formatUptime } from '@/lib/chat-utils'
import { GpuCard, DiskCard, ServerInfoCard } from '@/components/monitoring/SystemInfoCards'
import { Skeleton } from '@sloughgpt/strui'
import { apiPost } from '@/lib/http-client'
import { useErrorStore } from '@/lib/error-store'
import { activityController, type ActivityStatus } from '@/lib/activity-controller'
import { OutputCard } from '@/components/OutputCard'

export default function SystemHealthPage() {
  const [detailed, setDetailed] = useState<DetailedHealth | null>(null)
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [disk, setDisk] = useState<DiskUsage | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [knowledgeStats, setKnowledgeStats] = useState<{ total_items: number; topic_count: number; avg_importance: number; searchable: boolean } | null>(null)
  const [adapterStatus, setAdapterStatus] = useState<{ adapter_exists: boolean; fact_count: number; total_facts_available: number } | null>(null)
  const [benchQuality, setBenchQuality] = useState<{
    status: string; total_responses: number; coherence_score: number; quality_score: number;
    repetition_rate: number; avg_length: number; empty_rate: number;
  } | null>(null)
  const [benchStats, setBenchStats] = useState<{ total: number; avg_tokens: number; models: string[] } | null>(null)
  const [lastUpdated, setLastUpdated] = useState<string | null>(null)
  const [chartHistory, setChartHistory] = useState<Array<{ time: string; cpu: number; mem: number }>>([])
  const [dpoStatus, setDpoStatus] = useState<{ status: string; last_run: string | null; accepted_count: number; rejected_count: number; result: any } | null>(null)
  const [dpoRunning, setDpoRunning] = useState(false)
  const [visualStatus, setVisualStatus] = useState<{ visual_loaded: boolean; training: { status: string } } | null>(null)
  const [activityStatus, setActivityStatus] = useState<ActivityStatus | null>(null)
  const [activityTraining, setActivityTraining] = useState(false)
  const [activityTrainProgress, setActivityTrainProgress] = useState<string | null>(null)
  const MAX_HISTORY = 30
  const recentErrors = useErrorStore(s => s.errors)
  const dismissError = useErrorStore(s => s.dismissError)
  const clearErrors = useErrorStore(s => s.clearErrors)

  const fetchAll = useCallback(async (showRefreshing = false) => {
    if (showRefreshing) setRefreshing(true)
    setError(null)
    try {
      const [d, m, i, di, ks, as, bq, bs, dsRes, vs] = await Promise.all([
        systemController.getDetailedHealth().catch(() => null),
        systemController.getMetrics().catch(() => null),
        systemController.getInfo().catch(() => null),
        systemController.getDisk().catch(() => null),
        knowledgeController.stats().catch(() => null),
        knowledgeController.getAdapterStatus().catch(() => null),
        benchmarkController.quality().catch(() => null),
        benchmarkController.stats().catch(() => null),
        multimodalController.getDPOStatus().catch(() => null),
        multimodalController.getStatus().catch(() => null),
        activityController.status().catch(() => null),
      ])
      setDetailed(d)
      setMetrics(m)
      setInfo(i)
      setDisk(di)
      setKnowledgeStats(ks)
      setAdapterStatus(as)
      if (bq && 'coherence_score' in bq) {
        setBenchQuality(bq as any)
      } else {
        setBenchQuality(null)
      }
      setBenchStats(bs as any)
      setDpoStatus(dsRes as typeof dpoStatus)
      setVisualStatus(null)
      setLastUpdated(new Date().toLocaleTimeString())
      if (m) {
        setChartHistory(prev => {
          const next = [...prev, { time: new Date().toLocaleTimeString(), cpu: m.cpu_percent, mem: m.memory_percent }]
          if (next.length > MAX_HISTORY) next.shift()
          return next
        })
      }
    } catch (e: any) {
      setError(e?.message || 'Failed to load system health')
    }
    if (showRefreshing) setRefreshing(false)
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

  const loaded = detailed !== null
  const apiOk = detailed?.status === 'healthy'

  // Periodic polling — pauses when tab is hidden
  useEffect(() => {
    if (!loaded) return
    const poll = () => {
      if (document.hidden) return
      fetchAll()
    }
    const id = setInterval(poll, 5000)
    const handleVisibility = () => {
      if (!document.hidden) fetchAll(true)
    }
    document.addEventListener('visibilitychange', handleVisibility)
    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', handleVisibility)
    }
  }, [loaded, fetchAll])

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={
          <AppRouteHeaderLead
            title="System Health"
          />
        }
        right={
          <div className="flex items-center gap-2">
            {lastUpdated && (
              <span className="text-[11px] text-muted-foreground hidden sm:inline">Updated {lastUpdated}</span>
            )}
            <Button variant="outline" size="sm" onClick={() => fetchAll(true)} disabled={refreshing || !loaded}>
              {refreshing ? 'Refreshing...' : 'Refresh'}
            </Button>
          </div>
        }
      />
      <div className="space-y-4">
        {!loaded && (
          <Card>
            <CardHeader><CardTitle className="text-base">Status</CardTitle></CardHeader>
            <CardContent>
              <KpiGrid columns={4}>
                <StatCard label="API" value={<Skeleton className="h-4 w-16" />} />
                <StatCard label="Model" value={<Skeleton className="h-4 w-20" />} />
                <StatCard label="Uptime" value={<Skeleton className="h-4 w-16" />} />
                <StatCard label="Responses served" value={<Skeleton className="h-4 w-8" />} />
              </KpiGrid>
            </CardContent>
          </Card>
        )}

        {/* Status + Model */}
        {loaded && <Card>
          <CardHeader><CardTitle className="text-base">Status</CardTitle></CardHeader>
          <CardContent>
            <KpiGrid columns={4}>
              <StatCard
                label="API"
                value={!loaded ? 'Loading...' : apiOk ? 'Healthy' : 'Error'}
                icon={
                  <span className={`inline-block w-2 h-2 rounded-full ${!loaded ? 'bg-warning' : apiOk ? 'bg-success' : 'bg-destructive'}`} />
                }
              />
              <StatCard
                label="Model"
                value={!loaded ? '...' : detailed?.model_loaded ? (detailed.model_type || 'Loaded') : 'Not loaded'}
                icon={
                  <span className={`inline-block w-2 h-2 rounded-full ${!loaded ? 'bg-warning' : detailed?.model_loaded ? 'bg-success' : 'bg-warning'}`} />
                }
              />
              <StatCard
                label="Uptime"
                value={!loaded ? '...' : formatUptime(detailed.uptime_seconds)}
              />
              <StatCard
                label="Responses served"
                value={!loaded ? '...' : detailed?.inference?.inference_count ?? 0}
              />
            </KpiGrid>
          </CardContent>
        </Card>}

        {/* System Resources */}
        <Card>
          <CardHeader><CardTitle className="text-base">System Resources</CardTitle></CardHeader>
          <CardContent>
            <KpiGrid columns={4}>
              <StatCard
                label="CPU"
                value={metrics ? `${metrics.cpu_percent}%` : '...'}
                icon={
                  <span className={`inline-block w-2 h-2 rounded-full ${!metrics ? 'bg-warning' : metrics.cpu_percent > 80 ? 'bg-warning' : 'bg-success'}`} />
                }
              />
              <StatCard
                label="Memory"
                value={metrics ? `${metrics.memory_percent}%` : '...'}
                icon={
                  <span className={`inline-block w-2 h-2 rounded-full ${!metrics ? 'bg-warning' : metrics.memory_percent > 80 ? 'bg-warning' : 'bg-success'}`} />
                }
              />
              <StatCard
                label="Used"
                value={metrics ? `${metrics.memory_used_gb.toFixed(1)} GB` : '...'}
              />
              <StatCard
                label="Available"
                value={detailed?.system?.memory_available_mb ? `${(detailed.system.memory_available_mb / 1024).toFixed(1)} GB` : '...'}
              />
            </KpiGrid>
          </CardContent>
        </Card>

        {/* Knowledge Base */}
        <Card>
          <CardHeader><CardTitle className="text-base">Knowledge Base</CardTitle></CardHeader>
          <CardContent>
            <KpiGrid columns={4}>
              <StatCard label="Items" value={knowledgeStats ? knowledgeStats.total_items.toString() : '...'} />
              <StatCard label="Topics" value={knowledgeStats ? knowledgeStats.topic_count.toString() : '...'} />
              <StatCard label="Avg Importance" value={knowledgeStats ? knowledgeStats.avg_importance.toFixed(2) : '...'} />
              <StatCard
                label="AI training"
                value={!adapterStatus ? '...' : adapterStatus.adapter_exists ? 'Trained' : 'Not trained'}
                icon={
                  <span className={`inline-block w-2 h-2 rounded-full ${!adapterStatus ? 'bg-warning' : adapterStatus.adapter_exists ? 'bg-success' : 'bg-muted-foreground/50'}`} />
                }
              />
            </KpiGrid>
            {adapterStatus && adapterStatus.adapter_exists && (
              <p className="text-xs text-muted-foreground mt-2">
                Trained on {adapterStatus.fact_count} facts ({adapterStatus.total_facts_available} available)
              </p>
            )}
          </CardContent>
        </Card>

        {/* Model Quality */}
        {benchQuality && benchQuality.status === 'ok' && (
          <Card>
            <CardHeader><CardTitle className="text-base">Response quality</CardTitle></CardHeader>
            <CardContent>
              <KpiGrid columns={4}>
                <StatCard label="Coherence" value={benchQuality.coherence_score.toFixed(2)} icon={
                  <span className={`inline-block w-2 h-2 rounded-full ${benchQuality.coherence_score > 0.7 ? 'bg-success' : benchQuality.coherence_score > 0.4 ? 'bg-warning' : 'bg-destructive'}`} />
                } />
                <StatCard label="Quality Score" value={benchQuality.quality_score.toFixed(2)} icon={
                  <span className={`inline-block w-2 h-2 rounded-full ${benchQuality.quality_score > 0.7 ? 'bg-success' : benchQuality.quality_score > 0.4 ? 'bg-warning' : 'bg-destructive'}`} />
                } />
                <StatCard label="Responses" value={benchQuality.total_responses.toString()} />
                <StatCard label="Repetition" value={`${(benchQuality.repetition_rate * 100).toFixed(1)}%`} />
              </KpiGrid>
              <div className="flex gap-4 mt-2 text-xs text-muted-foreground">
                <span>Avg length: {benchQuality.avg_length.toFixed(1)} words</span>
                <span>Empty rate: {(benchQuality.empty_rate * 100).toFixed(1)}%</span>
                {benchStats && <span>Avg response length: {benchStats.avg_tokens.toFixed(0)}</span>}
              </div>
            </CardContent>
          </Card>
        )}

        {/* DPO / Visual Training */}
        {dpoStatus || visualStatus ? (
          <Card>
            <CardHeader><CardTitle className="text-base">Model Training (Feedback + Vision model)</CardTitle></CardHeader>
            <CardContent>
              <KpiGrid columns={4}>
                <StatCard
                  label="Training feedback status"
                  value={dpoStatus ? dpoStatus.status : '...'}
                  icon={
                    <span className={`inline-block w-2 h-2 rounded-full ${!dpoStatus ? 'bg-warning' : dpoStatus.status === 'running' ? 'bg-warning' : dpoStatus.status === 'completed' ? 'bg-success' : dpoStatus.status === 'error' ? 'bg-destructive' : 'bg-muted-foreground/50'}`}
                    />
                  }
                />
                <StatCard
                  label="Feedback accepted"
                  value={dpoStatus ? dpoStatus.accepted_count.toString() : '...'}
                />
                <StatCard
                  label="Feedback rejected"
                  value={dpoStatus ? dpoStatus.rejected_count.toString() : '...'}
                />
                <StatCard
                  label="Vision model loaded"
                  value={visualStatus ? (visualStatus.visual_loaded ? 'Yes' : 'No') : '...'}
                  icon={
                    <span className={`inline-block w-2 h-2 rounded-full ${!visualStatus ? 'bg-warning' : visualStatus.visual_loaded ? 'bg-success' : 'bg-muted-foreground/50'}`}
                    />
                  }
                />
              </KpiGrid>
              {visualStatus?.training && (
                <p className="text-xs text-muted-foreground mt-2">
                  Vision model training: {visualStatus.training.status}
                </p>
              )}
              {dpoStatus?.last_run && (
                <p className="text-xs text-muted-foreground mt-1">
                  Last feedback: {dpoStatus.last_run}
                </p>
              )}
              <div className="mt-3">
                <Button
                  size="sm"
                  disabled={dpoRunning || dpoStatus?.status === 'running'}
                  onClick={async () => {
                    setDpoRunning(true)
                    try {
                      await apiPost('/multimodal/dpo', {})
                      await fetchAll()
                    } catch {}
                    setDpoRunning(false)
                  }}
                  aria-label="Run feedback training"
                >
                  {dpoRunning || dpoStatus?.status === 'running' ? 'Running...' : 'Run feedback'}
                </Button>
              </div>
            </CardContent>
          </Card>
        ) : null}

        {/* Real-time chart */}
        {chartHistory.length > 1 && (
          <Card>
            <CardHeader><CardTitle className="text-base">Real‑time Metrics (last {MAX_HISTORY}s)</CardTitle></CardHeader>
            <CardContent>
              <div className="h-48" role="img" aria-label="CPU and memory usage chart over time">
                <SystemChart data={chartHistory} />
              </div>
            </CardContent>
          </Card>
        )}

        {/* GPU + Disk + Server */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <GpuCard gpu={detailed?.gpu as GPUInfo | undefined} />
          <DiskCard disk={disk ?? undefined} />
          <ServerInfoCard info={info ?? undefined} />
        </div>

        {/* Server Output */}
        <OutputCard />

        {recentErrors.length > 0 && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Recent Errors ({recentErrors.length})</CardTitle>
                <Button variant="ghost" size="sm" onClick={clearErrors}>
                  Clear
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-1 max-h-[240px] overflow-y-auto">
              {recentErrors.slice(0, 15).map(e => {
                const colors: Record<string, string> = { error: 'bg-destructive/10 border-destructive/30 text-destructive', warning: 'bg-warning/10 border-warning/30 text-warning', info: 'bg-muted border-border/60 text-muted-foreground' }
                return (
                  <div key={e.id} className={`flex items-start gap-2 p-2 rounded border text-xs ${colors[e.severity] || colors.error}`}>
                    <div className="flex-1 min-w-0">
                      <div className="font-medium truncate">{e.title}</div>
                      <div className="truncate opacity-80">{e.message}</div>
                      <div className="text-[10px] opacity-60 mt-0.5">
                        {new Date(e.timestamp).toLocaleTimeString()}
                        {e.source && <> · {e.source}</>}
                      </div>
                    </div>
                    {e.dismissible !== false && (
                      <button onClick={() => dismissError(e.id)} className="shrink-0 opacity-50 hover:opacity-100 text-xs leading-none" aria-label="Dismiss">&times;</button>
                    )}
                  </div>
                )
              })}
            </CardContent>
          </Card>
        )}

        {error && (
          <Card className="border-destructive/50">
            <CardContent className="py-3 text-sm text-destructive">
              {error}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
