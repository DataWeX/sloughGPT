'use client'
export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { systemController, type DetailedHealth, type SystemMetrics, type SystemInfo, type DiskUsage, type GPUInfo, type ExecutorStatus } from '@/lib/system-controller'
import { trainingController, type TrainingJob } from '@/lib/training-controller'
import { knowledgeController } from '@/lib/knowledge-controller'
import { benchmarkController } from '@/lib/benchmark-controller'
import { multimodalController } from '@/lib/controllers'
import type { AutoTrainStatus } from '@/lib/training-controller'
import dynamicNext from 'next/dynamic'
import { useLiveStatus } from '@/hooks/useLiveStatus'

const SystemChart = dynamicNext(() => import('@/components/monitoring/SystemChart').then(m => m.SystemChart), {
  ssr: false,
  loading: () => <div className="h-40 w-full animate-pulse bg-muted rounded-lg" />,
})
import { formatUptime } from '@/lib/chat-utils'
import { GpuCard, DiskCard, ServerInfoCard } from '@/components/monitoring/SystemInfoCards'
import { Skeleton } from '@sloughgpt/strui'
import { apiPost } from '@/lib/http-client'
import { ActivityTicker, ErrorList } from '@/components/ActivityTicker'
import { OutputCard } from '@/components/OutputCard'

export default function SystemHealthPage() {
  const { health: liveHealth, connectionStatus } = useLiveStatus()
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
  const [executorStatus, setExecutorStatus] = useState<ExecutorStatus | null>(null)
  const [autoTrainStatus, setAutoTrainStatus] = useState<AutoTrainStatus | null>(null)
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([])
  const MAX_HISTORY = 30

  const fetchAll = useCallback(async (showRefreshing = false) => {
    if (showRefreshing) setRefreshing(true)
    setError(null)
    try {
      const [d, m, i, di, ks, as, bq, bs, dsRes, vs, ex, at, tj] = await Promise.all([
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
        systemController.getExecutorStatus().catch(() => null),
        trainingController.getAutoTrainStatus().catch(() => null),
        trainingController.list().catch(() => []),
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
      setExecutorStatus(ex)
      setAutoTrainStatus(at)
      setTrainingJobs(Array.isArray(tj) ? tj : [])
      setLastUpdated(new Date().toLocaleTimeString())
    } catch (e: any) {
      setError(e?.message || 'Failed to load system health')
    }
    if (showRefreshing) setRefreshing(false)
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

  const loaded = detailed !== null
  const apiOk = (liveHealth?.health_status ?? detailed?.status) === 'healthy'

  // Live SSE data feeds the chart — no need for polling
  useEffect(() => {
    if (!liveHealth || !loaded) return
    setChartHistory(prev => {
      const next = [...prev, { time: new Date().toLocaleTimeString(), cpu: liveHealth.cpu_percent ?? 0, mem: liveHealth.memory_percent ?? 0 }]
      if (next.length > MAX_HISTORY) next.shift()
      return next
    })
  }, [liveHealth, loaded])

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

        {/* Status + Model — uses live SSE data for instant updates */}
        {loaded && <Card>
          <CardHeader><CardTitle className="text-base flex items-center gap-2">
            Status
            {connectionStatus === 'connected' && liveHealth && (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-success/10 text-success text-[10px] font-medium">
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
                live
              </span>
            )}
          </CardTitle></CardHeader>
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
                value={!loaded ? '...' : (liveHealth?.model_loaded ?? detailed?.model_loaded) ? (liveHealth?.model_type || detailed?.model_type || 'Loaded') : 'Not loaded'}
                icon={
                  <span className={`inline-block w-2 h-2 rounded-full ${!loaded ? 'bg-warning' : (liveHealth?.model_loaded ?? detailed?.model_loaded) ? 'bg-success' : 'bg-warning'}`} />
                }
              />
              <StatCard
                label="Uptime"
                value={!loaded ? '...' : formatUptime(liveHealth?.uptime_seconds ?? detailed?.uptime_seconds ?? 0)}
              />
              <StatCard
                label="Responses served"
                value={!loaded ? '...' : String(liveHealth?.inference_count ?? detailed?.inference?.inference_count ?? 0)}
              />
            </KpiGrid>
          </CardContent>
        </Card>}

        {/* System Resources — uses live SSE data for CPU/memory */}
        <Card>
          <CardHeader><CardTitle className="text-base">System Resources</CardTitle></CardHeader>
          <CardContent>
            <KpiGrid columns={4}>
              <StatCard
                label="CPU"
                value={liveHealth?.cpu_percent != null ? `${liveHealth.cpu_percent}%` : metrics ? `${metrics.cpu_percent}%` : '...'}
                icon={
                  <span className={`inline-block w-2 h-2 rounded-full ${(liveHealth?.cpu_percent ?? metrics?.cpu_percent ?? -1) < 0 ? 'bg-warning' : (liveHealth?.cpu_percent ?? metrics?.cpu_percent ?? 0) > 80 ? 'bg-warning' : 'bg-success'}`} />
                }
              />
              <StatCard
                label="Memory"
                value={liveHealth?.memory_percent != null ? `${liveHealth.memory_percent}%` : metrics ? `${metrics.memory_percent}%` : '...'}
                icon={
                  <span className={`inline-block w-2 h-2 rounded-full ${(liveHealth?.memory_percent ?? metrics?.memory_percent ?? -1) < 0 ? 'bg-warning' : (liveHealth?.memory_percent ?? metrics?.memory_percent ?? 0) > 80 ? 'bg-warning' : 'bg-success'}`} />
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

        {/* Auto-Trainer */}
        {autoTrainStatus && (
          <Card>
            <CardHeader><CardTitle className="text-base">Auto-Trainer</CardTitle></CardHeader>
            <CardContent>
              <KpiGrid columns={4}>
                <StatCard
                  label="Status"
                  value={autoTrainStatus.enabled ? 'Running' : 'Off'}
                  icon={
                    <span className={`inline-block w-2 h-2 rounded-full ${autoTrainStatus.enabled ? 'bg-success' : 'bg-muted-foreground/50'}`} />
                  }
                />
                <StatCard label="Conversations" value={`${autoTrainStatus.pending_conversations} / ${autoTrainStatus.threshold}`} />
                <StatCard label="Trains completed" value={autoTrainStatus.total_trains.toString()} />
                <StatCard
                  label="Last loss"
                  value={autoTrainStatus.last_loss != null ? autoTrainStatus.last_loss.toFixed(4) : '...'}
                />
              </KpiGrid>
              {autoTrainStatus.last_train && (
                <p className="text-xs text-muted-foreground mt-2">
                  Last trained: {new Date(autoTrainStatus.last_train).toLocaleString()}
                  {autoTrainStatus.last_checkpoint && <> · {autoTrainStatus.last_checkpoint}</>}
                </p>
              )}
              <p className="text-[11px] text-muted-foreground/50 mt-1">
                {autoTrainStatus.session_count} conversations · {autoTrainStatus.response_log_count} log files · interval {autoTrainStatus.interval_s}s
              </p>
            </CardContent>
          </Card>
        )}

        {/* Training History */}
        {trainingJobs.length > 0 && (
          <Card>
            <CardHeader><CardTitle className="text-base">Training History</CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-2">
                {trainingJobs.slice(0, 10).map((job) => (
                  <div key={job.id} className="flex items-center justify-between text-sm py-1 border-b last:border-0">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${
                        job.status === 'completed' ? 'bg-success' :
                        job.status === 'running' ? 'bg-primary animate-pulse' :
                        job.status === 'failed' ? 'bg-destructive' :
                        'bg-muted-foreground/50'
                      }`} />
                      <span className="truncate">{job.name || job.id}</span>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground shrink-0 ml-4">
                      {job.loss != null && <span>loss {job.loss.toFixed(3)}</span>}
                      {job.epochs_completed != null && <span>ep {job.epochs_completed}</span>}
                      <span>{job.status}</span>
                      <span>{new Date(job.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                ))}
              </div>
              {trainingJobs.length > 10 && (
                <p className="text-[11px] text-muted-foreground/50 mt-2">
                  + {trainingJobs.length - 10} more jobs
                </p>
              )}
            </CardContent>
          </Card>
        )}

        {/* Training Executor Pool */}
        {executorStatus && executorStatus.initialized && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center justify-between">
                <span>Training Pool</span>
                <div className="flex gap-2">
                  {executorStatus.total_tracked > 0 && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-[11px] h-6"
                      onClick={async () => {
                        await systemController.purgeExecutorJobs(3600)
                        fetchAll()
                      }}
                    >
                      Purge old
                    </Button>
                  )}
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <KpiGrid columns={4}>
                <StatCard
                  label="Active jobs"
                  value={executorStatus.active_jobs.toString()}
                  icon={
                    <span className={`inline-block w-2 h-2 rounded-full ${executorStatus.active_jobs > 0 ? 'bg-warning' : 'bg-success'}`} />
                  }
                />
                <StatCard label="Max workers" value={executorStatus.max_workers.toString()} />
                <StatCard label="Total tracked" value={executorStatus.total_tracked.toString()} />
                <StatCard
                  label="Queue"
                  value={executorStatus.jobs.filter(j => j.status === 'queued').length.toString()}
                />
              </KpiGrid>
              {executorStatus.jobs.length > 0 && (
                <div className="mt-3 text-xs text-muted-foreground space-y-1">
                  {executorStatus.jobs.slice(0, 5).map(j => {
                    const displayStatus = j.cancel_requested && j.status === 'running' ? 'cancelling' : j.status
                    return (
                    <div key={j.job_id} className="flex items-center gap-2">
                      <span className={`inline-block w-1.5 h-1.5 rounded-full ${
                        displayStatus === 'running' ? 'bg-warning' :
                        displayStatus === 'cancelling' ? 'bg-warning animate-pulse' :
                        displayStatus === 'completed' ? 'bg-success' :
                        displayStatus === 'failed' ? 'bg-destructive' :
                        displayStatus === 'cancelled' ? 'bg-muted-foreground/50' :
                        'bg-muted-foreground/30'
                      }`} />
                      <span className="font-mono">{j.job_id}</span>
                      <span className="text-muted-foreground/60">{displayStatus}</span>
                      {j.elapsed_s != null && <span>{j.elapsed_s.toFixed(1)}s</span>}
                      {j.tree_id && <span className="text-muted-foreground/40">tree:{j.tree_id}</span>}
                      {(j.status === 'running' || j.status === 'queued') && !j.cancel_requested && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-[10px] h-5 text-destructive hover:text-destructive ml-auto"
                          onClick={async () => {
                            await systemController.cancelExecutorJob(j.job_id)
                            fetchAll()
                          }}
                        >
                          Cancel
                        </Button>
                      )}
                    </div>
                    )
                  })}
                  {executorStatus.jobs.length > 5 && (
                    <div className="text-muted-foreground/40">+{executorStatus.jobs.length - 5} more</div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        )}

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
                    } catch (err) {
                      console.error('DPO training failed:', err)
                    }
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

        <ActivityTicker />

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Activity Log</CardTitle>
          </CardHeader>
          <CardContent className="max-h-[240px] overflow-y-auto">
            <ErrorList />
          </CardContent>
        </Card>

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
