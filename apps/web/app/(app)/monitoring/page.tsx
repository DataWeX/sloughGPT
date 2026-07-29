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
import { logger } from '@/lib/dev-log'
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
  const [dpoStatus, setDpoStatus] = useState<{ status: string; last_run: string | null; accepted_count: number; rejected_count: number; result: { perplexity_delta?: number; bleu_delta?: number; verdict?: string; report_path?: string } | null } | null>(null)
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
        setBenchQuality(bq as { status: string; total_responses: number; coherence_score: number; quality_score: number; repetition_rate: number; avg_length: number; empty_rate: number })
      } else {
        setBenchQuality(null)
      }
      setBenchStats(bs as { total: number; avg_tokens: number; models: string[] } | null)
      setDpoStatus(dsRes as typeof dpoStatus)
      setVisualStatus(null)
      setExecutorStatus(ex)
      setAutoTrainStatus(at)
      setTrainingJobs(Array.isArray(tj) ? tj : [])
      setLastUpdated(new Date().toLocaleTimeString())
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load system health')
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
              <span className="text-[11px] text-muted-foreground hidden sm:inline font-mono">Updated {lastUpdated}</span>
            )}
            <Button variant="outline" size="sm" onClick={() => fetchAll(true)} disabled={refreshing || !loaded}>
              {refreshing ? 'Refreshing...' : 'Refresh'}
            </Button>
          </div>
        }
      />
      <div className="space-y-4">
        {!loaded && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Card className="p-3"><CardContent className="p-0"><KpiGrid columns={4}>
              <StatCard label="API" value={<Skeleton className="h-4 w-16" />} />
              <StatCard label="Model" value={<Skeleton className="h-4 w-20" />} />
              <StatCard label="Uptime" value={<Skeleton className="h-4 w-16" />} />
              <StatCard label="Responses" value={<Skeleton className="h-4 w-8" />} />
            </KpiGrid></CardContent></Card>
            <Card className="p-3"><CardContent className="p-0"><KpiGrid columns={4}>
              <StatCard label="CPU" value={<Skeleton className="h-4 w-12" />} />
              <StatCard label="Memory" value={<Skeleton className="h-4 w-12" />} />
              <StatCard label="Used" value={<Skeleton className="h-4 w-16" />} />
              <StatCard label="Available" value={<Skeleton className="h-4 w-16" />} />
            </KpiGrid></CardContent></Card>
            <Card className="p-3"><CardContent className="p-0"><KpiGrid columns={4}>
              <StatCard label="Items" value={<Skeleton className="h-4 w-8" />} />
              <StatCard label="Topics" value={<Skeleton className="h-4 w-8" />} />
              <StatCard label="Importance" value={<Skeleton className="h-4 w-12" />} />
              <StatCard label="AI training" value={<Skeleton className="h-4 w-16" />} />
            </KpiGrid></CardContent></Card>
          </div>
        )}

        {/* Row 1: Status + Resources + Knowledge — 3-col grid */}
        {loaded && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Card className="p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Status</span>
                {connectionStatus === 'connected' && liveHealth && (
                  <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-success/10 text-success text-[10px] font-medium">
                    <span className="inline-block h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
                    live
                  </span>
                )}
              </div>
              <CardContent className="p-0">
                <KpiGrid columns={2}>
                  <StatCard
                    label="API"
                    value={!loaded ? '...' : <span className="font-mono">{apiOk ? 'Healthy' : 'Error'}</span>}
                    icon={<span className={`inline-block w-2 h-2 rounded-full ${!loaded ? 'bg-warning' : apiOk ? 'bg-success' : 'bg-destructive'}`} />}
                  />
                  <StatCard
                    label="Model"
                    value={!loaded ? '...' : <span className="font-mono">{(liveHealth?.model_loaded ?? detailed?.model_loaded) ? (liveHealth?.model_type || detailed?.model_type || 'Loaded') : 'Not loaded'}</span>}
                    icon={<span className={`inline-block w-2 h-2 rounded-full ${!loaded ? 'bg-warning' : (liveHealth?.model_loaded ?? detailed?.model_loaded) ? 'bg-success' : 'bg-warning'}`} />}
                  />
                  <StatCard
                    label="Uptime"
                    value={!loaded ? '...' : <span className="font-mono">{formatUptime(liveHealth?.uptime_seconds ?? detailed?.uptime_seconds ?? 0)}</span>}
                  />
                  <StatCard
                    label="Responses"
                    value={!loaded ? '...' : <span className="font-mono">{String(liveHealth?.inference_count ?? detailed?.inference?.inference_count ?? 0)}</span>}
                  />
                </KpiGrid>
              </CardContent>
            </Card>

            <Card className="p-3">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Resources</span>
              <CardContent className="p-0">
                <KpiGrid columns={2}>
                  <StatCard
                    label="CPU"
                    value={liveHealth?.cpu_percent != null ? <span className="font-mono">{liveHealth.cpu_percent}%</span> : metrics ? <span className="font-mono">{metrics.cpu_percent}%</span> : '...'}
                    icon={<span className={`inline-block w-2 h-2 rounded-full ${(liveHealth?.cpu_percent ?? metrics?.cpu_percent ?? -1) < 0 ? 'bg-warning' : (liveHealth?.cpu_percent ?? metrics?.cpu_percent ?? 0) > 80 ? 'bg-warning' : 'bg-success'}`} />}
                  />
                  <StatCard
                    label="Memory"
                    value={liveHealth?.memory_percent != null ? <span className="font-mono">{liveHealth.memory_percent}%</span> : metrics ? <span className="font-mono">{metrics.memory_percent}%</span> : '...'}
                    icon={<span className={`inline-block w-2 h-2 rounded-full ${(liveHealth?.memory_percent ?? metrics?.memory_percent ?? -1) < 0 ? 'bg-warning' : (liveHealth?.memory_percent ?? metrics?.memory_percent ?? 0) > 80 ? 'bg-warning' : 'bg-success'}`} />}
                  />
                  <StatCard
                    label="Used"
                    value={metrics ? <span className="font-mono">{metrics.memory_used_gb.toFixed(1)} GB</span> : '...'}
                  />
                  <StatCard
                    label="Available"
                    value={detailed?.system?.memory_available_mb ? <span className="font-mono">{(detailed.system.memory_available_mb / 1024).toFixed(1)} GB</span> : '...'}
                  />
                </KpiGrid>
              </CardContent>
            </Card>

            <Card className="p-3">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Knowledge</span>
              <CardContent className="p-0">
                <KpiGrid columns={2}>
                  <StatCard label="Items" value={knowledgeStats ? <span className="font-mono">{knowledgeStats.total_items.toString()}</span> : '...'} />
                  <StatCard label="Topics" value={knowledgeStats ? <span className="font-mono">{knowledgeStats.topic_count.toString()}</span> : '...'} />
                  <StatCard label="Importance" value={knowledgeStats ? <span className="font-mono">{knowledgeStats.avg_importance.toFixed(2)}</span> : '...'} />
                  <StatCard
                    label="AI training"
                    value={!adapterStatus ? '...' : <span className="font-mono">{adapterStatus.adapter_exists ? 'Trained' : 'Not'}</span>}
                    icon={<span className={`inline-block w-2 h-2 rounded-full ${!adapterStatus ? 'bg-warning' : adapterStatus.adapter_exists ? 'bg-success' : 'bg-muted-foreground/50'}`} />}
                  />
                </KpiGrid>
                {adapterStatus && adapterStatus.adapter_exists && (
                  <p className="text-[11px] text-muted-foreground mt-1.5 font-mono">
                    {adapterStatus.fact_count} facts ({adapterStatus.total_facts_available} avail)
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {/* Row 2: Auto-Trainer + Quality + DPO — 3-col grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {autoTrainStatus && (
            <Card className="p-3">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Auto-Trainer</span>
              <CardContent className="p-0">
                <KpiGrid columns={2}>
                  <StatCard
                    label="Status"
                    value={autoTrainStatus.enabled ? <span className="font-mono">Running</span> : <span className="font-mono">Off</span>}
                    icon={<span className={`inline-block w-2 h-2 rounded-full ${autoTrainStatus.enabled ? 'bg-success' : 'bg-muted-foreground/50'}`} />}
                  />
                  <StatCard label="Queue" value={<span className="font-mono">{autoTrainStatus.pending_conversations}/{autoTrainStatus.threshold}</span>} />
                  <StatCard label="Trains" value={<span className="font-mono">{autoTrainStatus.total_trains.toString()}</span>} />
                  <StatCard label="Loss" value={autoTrainStatus.last_loss != null ? <span className="font-mono">{autoTrainStatus.last_loss.toFixed(4)}</span> : '...'} />
                </KpiGrid>
                {autoTrainStatus.last_train && (
                  <p className="text-[11px] text-muted-foreground mt-1.5 font-mono">
                    Last: {new Date(autoTrainStatus.last_train).toLocaleString()}
                    {autoTrainStatus.last_checkpoint && <> · {autoTrainStatus.last_checkpoint}</>}
                  </p>
                )}
                <p className="text-[10px] text-muted-foreground/50 mt-0.5 font-mono">
                  {autoTrainStatus.session_count} sessions · {autoTrainStatus.response_log_count} logs · {autoTrainStatus.interval_s}s
                </p>
              </CardContent>
            </Card>
          )}

          {benchQuality && benchQuality.status === 'ok' && (
            <Card className="p-3">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Quality</span>
              <CardContent className="p-0">
                <KpiGrid columns={2}>
                  <StatCard label="Coherence" value={<span className="font-mono">{benchQuality.coherence_score.toFixed(2)}</span>} icon={
                    <span className={`inline-block w-2 h-2 rounded-full ${benchQuality.coherence_score > 0.7 ? 'bg-success' : benchQuality.coherence_score > 0.4 ? 'bg-warning' : 'bg-destructive'}`} />
                  } />
                  <StatCard label="Score" value={<span className="font-mono">{benchQuality.quality_score.toFixed(2)}</span>} icon={
                    <span className={`inline-block w-2 h-2 rounded-full ${benchQuality.quality_score > 0.7 ? 'bg-success' : benchQuality.quality_score > 0.4 ? 'bg-warning' : 'bg-destructive'}`} />
                  } />
                  <StatCard label="Responses" value={<span className="font-mono">{benchQuality.total_responses.toString()}</span>} />
                  <StatCard label="Repetition" value={<span className="font-mono">{(benchQuality.repetition_rate * 100).toFixed(1)}%</span>} />
                </KpiGrid>
                <div className="flex gap-3 mt-1.5 text-[11px] text-muted-foreground font-mono">
                  <span>Avg: {benchQuality.avg_length.toFixed(1)}w</span>
                  <span>Empty: {(benchQuality.empty_rate * 100).toFixed(1)}%</span>
                  {benchStats && <span>Tokens: {benchStats.avg_tokens.toFixed(0)}</span>}
                </div>
              </CardContent>
            </Card>
          )}

          {(dpoStatus || visualStatus) && (
            <Card className="p-3">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Feedback + Vision</span>
              <CardContent className="p-0">
                <KpiGrid columns={2}>
                  <StatCard
                    label="Feedback"
                    value={dpoStatus ? <span className="font-mono">{dpoStatus.status}</span> : '...'}
                    icon={<span className={`inline-block w-2 h-2 rounded-full ${!dpoStatus ? 'bg-warning' : dpoStatus.status === 'running' ? 'bg-warning' : dpoStatus.status === 'completed' ? 'bg-success' : dpoStatus.status === 'error' ? 'bg-destructive' : 'bg-muted-foreground/50'}`} />}
                  />
                  <StatCard
                    label="Vision"
                    value={visualStatus ? <span className="font-mono">{visualStatus.visual_loaded ? 'Yes' : 'No'}</span> : '...'}
                    icon={<span className={`inline-block w-2 h-2 rounded-full ${!visualStatus ? 'bg-warning' : visualStatus.visual_loaded ? 'bg-success' : 'bg-muted-foreground/50'}`} />}
                  />
                  <StatCard label="Accepted" value={dpoStatus ? <span className="font-mono">{dpoStatus.accepted_count.toString()}</span> : '...'} />
                  <StatCard label="Rejected" value={dpoStatus ? <span className="font-mono">{dpoStatus.rejected_count.toString()}</span> : '...'} />
                </KpiGrid>
                <div className="mt-2">
                  <Button
                    size="sm"
                    className="h-6 text-[11px]"
                    disabled={dpoRunning || dpoStatus?.status === 'running'}
                    onClick={async () => {
                      setDpoRunning(true)
                      try {
                        await apiPost('/multimodal/dpo', {})
                        await fetchAll()
                      } catch (err) {
                        logger.error('DPO training failed', { exception: String(err) })
                      }
                      setDpoRunning(false)
                    }}
                  >
                    {dpoRunning || dpoStatus?.status === 'running' ? 'Running...' : 'Run feedback'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Row 3: Training History + Training Pool — 2-col grid */}
        {(trainingJobs.length > 0 || (executorStatus && executorStatus.initialized)) && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {trainingJobs.length > 0 && (
              <Card className="p-3">
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Training History</span>
                <CardContent className="p-0">
                  <div className="space-y-1">
                    {trainingJobs.slice(0, 6).map((job) => (
                      <div key={job.id} className="flex items-center justify-between text-xs py-1 px-1.5 rounded hover:bg-muted/30 transition-colors">
                        <div className="flex items-center gap-1.5 min-w-0">
                          <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                            job.status === 'completed' ? 'bg-success/15 text-success' :
                            job.status === 'running' ? 'bg-warning/15 text-warning' :
                            job.status === 'failed' ? 'bg-destructive/15 text-destructive' :
                            'bg-muted text-muted-foreground'
                          }`}>{job.status}</span>
                          <span className="truncate font-mono text-muted-foreground">{job.name || job.id}</span>
                        </div>
                        <div className="flex items-center gap-2 text-[10px] text-muted-foreground shrink-0 ml-2 font-mono">
                          {job.loss != null && <span>{job.loss.toFixed(3)}</span>}
                          {job.epochs_completed != null && <span>ep{job.epochs_completed}</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                  {trainingJobs.length > 6 && (
                    <p className="text-[10px] text-muted-foreground/50 mt-1 font-mono">+{trainingJobs.length - 6} more</p>
                  )}
                </CardContent>
              </Card>
            )}

            {executorStatus && executorStatus.initialized && (
              <Card className="p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Pool</span>
                  {executorStatus.total_tracked > 0 && (
                    <Button variant="outline" size="sm" className="text-[10px] h-7" onClick={async () => { await systemController.purgeExecutorJobs(3600); fetchAll() }}>
                      Purge
                    </Button>
                  )}
                </div>
                <CardContent className="p-0">
                  <KpiGrid columns={4}>
                    <StatCard
                      label="Active"
                      value={<span className="font-mono">{executorStatus.active_jobs.toString()}</span>}
                      icon={<span className={`inline-block w-2 h-2 rounded-full ${executorStatus.active_jobs > 0 ? 'bg-warning' : 'bg-success'}`} />}
                    />
                    <StatCard label="Workers" value={<span className="font-mono">{executorStatus.max_workers.toString()}</span>} />
                    <StatCard label="Tracked" value={<span className="font-mono">{executorStatus.total_tracked.toString()}</span>} />
                    <StatCard label="Queue" value={<span className="font-mono">{executorStatus.jobs.filter(j => j.status === 'queued').length.toString()}</span>} />
                  </KpiGrid>
                  {executorStatus.jobs.length > 0 && (
                    <div className="mt-2 text-[11px] text-muted-foreground space-y-0.5 font-mono">
                      {executorStatus.jobs.slice(0, 4).map(j => {
                        const ds = j.cancel_requested && j.status === 'running' ? 'cancelling' : j.status
                        return (
                        <div key={j.job_id} className="flex items-center gap-1.5 px-1 py-0.5 rounded hover:bg-muted/30 transition-colors">
                          <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                            ds === 'running' ? 'bg-warning/15 text-warning' : ds === 'cancelling' ? 'bg-warning/15 text-warning' :
                            ds === 'completed' ? 'bg-success/15 text-success' : ds === 'failed' ? 'bg-destructive/15 text-destructive' : 'bg-muted text-muted-foreground'
                          }`}>{ds}</span>
                          <span className="truncate">{j.job_id}</span>
                          {j.elapsed_s != null && <span className="text-muted-foreground/60">{j.elapsed_s.toFixed(1)}s</span>}
                          {(j.status === 'running' || j.status === 'queued') && !j.cancel_requested && (
                            <Button variant="ghost" size="sm" className="text-[10px] h-7 text-destructive hover:text-destructive ml-auto"
                              onClick={async () => { await systemController.cancelExecutorJob(j.job_id); fetchAll() }}>
                              Cancel
                            </Button>
                          )}
                        </div>
                        )
                      })}
                      {executorStatus.jobs.length > 4 && <div className="text-muted-foreground/40">+{executorStatus.jobs.length - 4} more</div>}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {/* Row 4: Chart */}
        {chartHistory.length > 1 && (
          <Card className="p-3">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Real-time (last {MAX_HISTORY}s)</span>
            <CardContent className="p-0">
              <div className="h-40" role="img" aria-label="CPU and memory usage chart over time">
                <SystemChart data={chartHistory} />
              </div>
            </CardContent>
          </Card>
        )}

        {/* Row 5: GPU + Disk + Server */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <GpuCard gpu={detailed?.gpu as GPUInfo | undefined} />
          <DiskCard disk={disk ?? undefined} />
          <ServerInfoCard info={info ?? undefined} />
        </div>

        {/* Row 6: Server Output + Activity */}
        <OutputCard compact />

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="md:col-span-2">
            <ActivityTicker />
          </div>
          <Card className="p-3">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Errors</span>
            <CardContent className="p-0 max-h-[200px] overflow-y-auto">
              <ErrorList />
            </CardContent>
          </Card>
        </div>

        {error && (
          <Card className="p-3 border-destructive/50">
            <CardContent className="p-0 py-2 text-sm text-destructive">{error}</CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
