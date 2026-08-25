'use client'
export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback, useRef, memo } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, FoldSection } from '@sloughgpt/strui'
import { Button, Switch } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { extractErrorMessage } from '@/lib/error-utils'
import { systemController, type DetailedHealth, type SystemMetrics, type SystemInfo, type DiskUsage, type GPUInfo, type ExecutorStatus } from '@/lib/system-controller'
import { trainingController, type TrainingJob } from '@/lib/training-controller'
import { knowledgeController } from '@/lib/knowledge-controller'
import { benchmarkController } from '@/lib/benchmark-controller'
import { multimodalController } from '@/lib/controllers'
import type { AutoTrainStatus } from '@/lib/training-controller'
import dynamicNext from 'next/dynamic'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { downloadJson } from '@/lib/download-utils'
import { todayDateString, getJsonItem } from '@/lib/format-bytes'
import { StatusCard } from '@/components/monitoring/StatusCard'
import { logger } from '@/lib/dev-log'
import { DiagnosticsCard } from '@/components/monitoring/DiagnosticsCard'
import { TrafficCard } from '@/components/monitoring/TrafficCard'
import { ModelMetricsCard } from '@/components/monitoring/ModelMetricsCard'
import { PathLatenciesCard } from '@/components/monitoring/PathLatenciesCard'
import { ServerErrorsCard } from '@/components/monitoring/ServerErrorsCard'
import { ModelEventsCard } from '@/components/monitoring/ModelEventsCard'
import { RateViolationsCard } from '@/components/monitoring/RateViolationsCard'
import { ResourceCard } from '@/components/monitoring/ResourceCard'
import { LatencyCard } from '@/components/monitoring/LatencyCard'
import { AlertPanel } from '@/components/monitoring/AlertPanel'
import { KnowledgeCard } from '@/components/monitoring/KnowledgeCard'
import { AutoTrainCard } from '@/components/monitoring/AutoTrainCard'
import { QualityCard } from '@/components/monitoring/QualityCard'
import { FeedbackCard } from '@/components/monitoring/FeedbackCard'
import { TrainingHistory } from '@/components/monitoring/TrainingHistory'
import { ExecutorPool } from '@/components/monitoring/ExecutorPool'
import { InferencePoolCard } from '@/components/monitoring/InferencePoolCard'
import { ProcessCard } from '@/components/monitoring/ProcessCard'
import { KvCacheCard } from '@/components/monitoring/KvCacheCard'
import { GpuCard, DiskCard, ServerInfoCard } from '@/components/monitoring/SystemInfoCards'
import { WorkflowCard } from '@/components/monitoring/WorkflowCard'
import { ActivityTicker, ErrorList } from '@/components/ActivityTicker'
import { OutputCard } from '@/components/OutputCard'

const POLL_INTERVAL_MS = 10_000
const POLL_MAX_BACKOFF_MS = 60_000
const MAX_ALERT_HISTORY = 20

const SystemChart = dynamicNext(() => import('@/components/monitoring/SystemChart').then(m => m.SystemChart), {
  ssr: false,
  loading: () => <div className="h-40 w-full animate-pulse bg-muted rounded-lg" />,
})

const TrendChart = dynamicNext(() => import('@/components/monitoring/TrendChart').then(m => m.TrendChart), {
  ssr: false,
  loading: () => <div className="h-40 w-full animate-pulse bg-muted rounded-lg" />,
})

export default memo(function SystemHealthPage() {
  const { health: liveHealth, connectionStatus } = useLiveStatus()
  const [detailed, setDetailed] = useState<DetailedHealth | null>(null)
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [disk, setDisk] = useState<DiskUsage | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [knowledgeStats, setKnowledgeStats] = useState<{ total_items: number; topic_count: number; avg_importance: number; searchable: boolean } | null>(null)
  const [adapterStatus, setAdapterStatus] = useState<{ adapter_exists: boolean; fact_count: number; total_facts_available: number } | null>(null)
  const [benchQuality, setBenchQuality] = useState<{
    coherence_score: number; quality_score: number; repetition_rate: number;
    total_responses: number; avg_length: number; empty_rate: number;
  } | null>(null)
  const [benchStats, setBenchStats] = useState<{ total: number; avg_tokens: number; models: string[] } | null>(null)
  const [lastUpdated, setLastUpdated] = useState<string | null>(null)
  const [chartHistory, setChartHistory] = useState<Array<{ time: string; cpu: number; mem: number; tokens?: number; latency?: number }>>([])
  const [dpoStatus, setDpoStatus] = useState<{ status: string; last_run: string | null; accepted_count: number; rejected_count: number; result: { perplexity_delta?: number; bleu_delta?: number; verdict?: string; report_path?: string } | null } | null>(null)
  const [dpoRunning, setDpoRunning] = useState(false)
  const [visualStatus, setVisualStatus] = useState<{ visual_loaded: boolean; training: { status: string } } | null>(null)
  const [executorStatus, setExecutorStatus] = useState<ExecutorStatus | null>(null)
  const [autoTrainStatus, setAutoTrainStatus] = useState<AutoTrainStatus | null>(null)
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([])
  const MAX_HISTORY = 30
  const [inferenceRate, setInferenceRate] = useState<number>(0)
  const prevInferenceRef = useRef<{ count: number; time: number } | null>(null)
  const rateEmaRef = useRef<number>(0)
  const [cpuThreshold, setCpuThreshold] = useState(() => {
    return getJsonItem<Record<string, number>>('sloughgpt-monitoring-thresholds', {}).cpu ?? 80
  })
  const [memThreshold, setMemThreshold] = useState(() => {
    return getJsonItem<Record<string, number>>('sloughgpt-monitoring-thresholds', {}).mem ?? 80
  })
  const [alerts, setAlerts] = useState<Array<{ time: string; type: string; value: number }>>([])
  const alertsRef = useRef<Array<{ time: string; type: string; value: number }>>([])
  const prevCpuOverRef = useRef(false)
  const prevMemOverRef = useRef(false)

  useEffect(() => {
    try { localStorage.setItem('sloughgpt-monitoring-thresholds', JSON.stringify({ cpu: cpuThreshold, mem: memThreshold })) } catch { /* ignore */ }
  }, [cpuThreshold, memThreshold])

  const fetchAll = useCallback(async (showRefreshing = false): Promise<boolean> => {
    if (showRefreshing) setRefreshing(true)
    setError(null)
    try {
      // Critical endpoint — failure means the page cannot render.
      const d = await systemController.getDetailedHealth()
      setDetailed(d)

      // Non-critical endpoints — each degrades independently on failure.
      const [m, i, di, ks, as_, bq, bs, dsRes, vs, ex, at, tj] = await Promise.all([
        systemController.getMetrics().catch((e) => { logger.warning('Could not metrics fetch', e); return null }),
        systemController.getInfo().catch((e) => { logger.warning('Could not info fetch', e); return null }),
        systemController.getDisk().catch((e) => { logger.warning('Could not disk fetch', e); return null }),
        knowledgeController.stats().catch((e) => { logger.warning('Could not knowledge stats', e); return null }),
        knowledgeController.getAdapterStatus().catch((e) => { logger.warning('Could not adapter status', e); return null }),
        benchmarkController.quality().catch((e) => { logger.warning('Could not benchmark quality', e); return null }),
        benchmarkController.stats().catch((e) => { logger.warning('Could not benchmark stats', e); return null }),
        multimodalController.getDPOStatus().catch((e) => { logger.warning('Could not dpo status', e); return null }),
        multimodalController.getStatus().catch((e) => { logger.warning('Could not multimodal status', e); return null }),
        systemController.getExecutorStatus().catch((e) => { logger.warning('Could not executor status', e); return null }),
        trainingController.getAutoTrainStatus().catch((e) => { logger.warning('auto-train status failed', e); return null }),
        trainingController.list().catch((e) => { logger.warning('Could not training list', e); return [] }),
      ])
      if (m != null) setMetrics(m)
      if (i != null) setInfo(i)
      if (di != null) setDisk(di)
      if (ks != null) setKnowledgeStats(ks)
      if (as_ != null) setAdapterStatus(as_)
      if (bq && 'coherence_score' in bq) {
        setBenchQuality(bq as { coherence_score: number; quality_score: number; repetition_rate: number; total_responses: number; avg_length: number; empty_rate: number })
      }
      if (bs != null) setBenchStats(bs as { total: number; avg_tokens: number; models: string[] } | null)
      if (dsRes != null) setDpoStatus(dsRes as typeof dpoStatus)
      if (vs != null) setVisualStatus({
        visual_loaded: vs.engine.vision_model != null,
        training: { status: vs.engine.status },
      })
      if (ex != null) setExecutorStatus(ex)
      if (at != null) setAutoTrainStatus(at)
      if (Array.isArray(tj)) setTrainingJobs(tj)
      setLastUpdated(new Date().toLocaleTimeString())
      return true
    } catch (e: unknown) {
      setError(extractErrorMessage(e, 'Could not load system health'))
      return false
    } finally {
      if (showRefreshing) setRefreshing(false)
    }
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void fetchAll(true) }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [fetchAll])

  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission()
    }
  }, [])

  const loaded = detailed !== null

  useEffect(() => {
    if (!liveHealth || !loaded) return
    const cpu = liveHealth.cpu_percent ?? 0
    const mem = liveHealth.memory_percent ?? 0
    const now = new Date().toLocaleTimeString()
    setLastUpdated(now)
    const cpuOver = cpu > cpuThreshold
    if (cpuOver && !prevCpuOverRef.current) {
      const alert = { time: now, type: 'CPU', value: cpu }
      alertsRef.current = [alert, ...alertsRef.current.slice(0, MAX_ALERT_HISTORY - 1)]
      setAlerts(alertsRef.current)
      if (Notification.permission === 'granted') {
        new Notification('High CPU Usage', { body: `CPU at ${cpu.toFixed(0)}% (threshold: ${cpuThreshold}%)`, icon: '/favicon.svg' })
      }
    }
    prevCpuOverRef.current = cpuOver
    const memOver = mem > memThreshold
    if (memOver && !prevMemOverRef.current) {
      const alert = { time: now, type: 'Memory', value: mem }
      alertsRef.current = [alert, ...alertsRef.current.slice(0, MAX_ALERT_HISTORY - 1)]
      setAlerts(alertsRef.current)
      if (Notification.permission === 'granted') {
        new Notification('High Memory Usage', { body: `Memory at ${mem.toFixed(0)}% (threshold: ${memThreshold}%)`, icon: '/favicon.svg' })
      }
    }
    prevMemOverRef.current = memOver
    setChartHistory(prev => {
      const next = [...prev, {
        time: now,
        cpu,
        mem,
        tokens: liveHealth.tokens_per_sec ?? 0,
        latency: liveHealth.avg_latency_ms ?? 0,
      }]
      if (next.length > MAX_HISTORY) next.shift()
      return next
    })
  }, [liveHealth, loaded, cpuThreshold, memThreshold])

  useEffect(() => {
    if (!liveHealth) return
    const now = Date.now()
    const count = liveHealth.inference_count ?? 0
    const prev = prevInferenceRef.current
    if (prev) {
      const elapsed = (now - prev.time) / 1000
      if (elapsed > 0) {
        const rawRate = ((count - prev.count) / elapsed) * 60
        // Exponential moving average (alpha=0.3) to smooth jitter
        const alpha = 0.3
        rateEmaRef.current = alpha * rawRate + (1 - alpha) * rateEmaRef.current
        setInferenceRate(Math.max(0, rateEmaRef.current))
      }
    }
    prevInferenceRef.current = { count, time: now }
  }, [liveHealth])

  useEffect(() => {
    if (!loaded || !autoRefresh) return
    if (connectionStatus !== 'connected') return
    let failures = 0
    let timer: ReturnType<typeof setTimeout> | null = null
    let generation = 0
    let fetching = false

    const run = async (gen: number) => {
      if (fetching) return
      if (document.hidden) return
      fetching = true
      const ok = await fetchAll()
      fetching = false
      if (gen !== generation) return
      failures = ok ? 0 : failures + 1
      const delay = Math.min(POLL_INTERVAL_MS * Math.pow(2, failures), POLL_MAX_BACKOFF_MS)
      timer = setTimeout(() => tick(), delay)
    }

    const tick = () => {
      const gen = ++generation
      if (timer !== null) { clearTimeout(timer); timer = null }
      run(gen)
    }

    const onVisible = () => {
      if (document.hidden || !autoRefresh) return
      failures = 0
      tick()
    }

    document.addEventListener('visibilitychange', onVisible)
    run(0)

    return () => {
      generation++
      if (timer !== null) clearTimeout(timer)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [loaded, fetchAll, autoRefresh, connectionStatus])

  const handleExportReport = () => {
    const report = {
      timestamp: new Date().toISOString(),
      api_health: liveHealth,
      connection_status: connectionStatus,
      detailed_health: detailed,
      system_metrics: metrics,
      system_info: info,
      disk_usage: disk,
      knowledge_stats: knowledgeStats,
      adapter_status: adapterStatus,
      benchmark_quality: benchQuality,
      training_jobs: trainingJobs,
      dpo_status: dpoStatus,
      executor_status: executorStatus,
      visual_status: visualStatus,
    }
    downloadJson(report, `system-report-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`)
  }

  const handleExportHistory = () => {
    if (chartHistory.length === 0) return
    downloadJson(chartHistory, `system-history-${todayDateString()}.json`)
  }

  const headerRight = (
    <div className="flex items-center gap-3">
      {lastUpdated && (
        <span className="text-[11px] text-muted-foreground hidden sm:inline font-mono" aria-live="polite" aria-atomic="true">Updated {lastUpdated}</span>
      )}
      <div className="flex items-center gap-1.5">
        <label className="text-[10px] text-muted-foreground">Auto</label>
        <Switch checked={autoRefresh} onCheckedChange={setAutoRefresh} className="scale-75" />
      </div>
      <Button variant="outline" size="sm" onClick={handleExportReport} disabled={!loaded}>
        Export
      </Button>
      <Button variant="outline" size="sm" onClick={() => fetchAll(true)} disabled={refreshing || !loaded}>
        {refreshing ? 'Refreshing...' : 'Refresh'}
      </Button>
    </div>
  )

  return (
    <PageContainer
      title="System Health"
      headerRight={headerRight}
    >
      {/* Loading: skeleton while fetch is in progress */}
      {!loaded && !error && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3" aria-busy="true">
          <Card className="p-4"><CardContent className="p-0"><div className="grid grid-cols-2 gap-3">
            {[1,2,3,4].map(i => <div key={i} className="space-y-1"><Skeleton className="h-3 w-12" /><Skeleton className="h-5 w-16" /></div>)}
          </div></CardContent></Card>
          <Card className="p-4"><CardContent className="p-0"><div className="grid grid-cols-2 gap-3">
            {[1,2,3,4].map(i => <div key={i} className="space-y-1"><Skeleton className="h-3 w-12" /><Skeleton className="h-5 w-16" /></div>)}
          </div></CardContent></Card>
          <Card className="p-4"><CardContent className="p-0"><div className="grid grid-cols-2 gap-3">
            {[1,2,3,4].map(i => <div key={i} className="space-y-1"><Skeleton className="h-3 w-12" /><Skeleton className="h-5 w-16" /></div>)}
          </div></CardContent></Card>
        </div>
      )}

      {/* Error: fetch failed — show message with retry */}
      {!loaded && error && (
        <Card className="p-6">
          <CardContent className="p-0 flex flex-col items-center gap-3 text-center">
            <p className="text-sm text-destructive">{error}</p>
            <Button size="sm" variant="outline" onClick={() => fetchAll(true)} disabled={refreshing}>
              {refreshing ? 'Retrying...' : 'Retry'}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Row 1: Status + Resources + Alerts — essential overview */}
      {loaded && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <StatusCard
            liveHealth={liveHealth}
            detailed={detailed}
            connectionStatus={connectionStatus}
            inferenceRate={inferenceRate}
            loaded={loaded}
          />
          <ResourceCard
            liveHealth={liveHealth}
            metrics={metrics}
            detailed={detailed}
            cpuThreshold={cpuThreshold}
            memThreshold={memThreshold}
            loaded={loaded}
          />
          <AlertPanel
            cpuThreshold={cpuThreshold}
            memThreshold={memThreshold}
            onCpuThresholdChange={setCpuThreshold}
            onMemThresholdChange={setMemThreshold}
            alerts={alerts}
          />
        </div>
      )}

      {/* Row 2: Server errors — always visible */}
      {loaded && <ServerErrorsCard liveHealth={liveHealth} />}

      {/* Row 3: Health & memory trend — always visible */}
      {loaded && (
        <Card className="p-4">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Health &amp; memory trend</span>
          <CardContent className="p-0">
            <div className="h-40">
              <TrendChart liveHealth={liveHealth} />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Collapsible: Diagnostics (SSE-driven real-time data) */}
      {loaded && (
        <FoldSection heading="Diagnostics">
          <div className="space-y-3">
            <DiagnosticsCard liveHealth={liveHealth} />
            <TrafficCard liveHealth={liveHealth} />
            <ModelMetricsCard liveHealth={liveHealth} />
            <PathLatenciesCard liveHealth={liveHealth} />
            <ModelEventsCard liveHealth={liveHealth} />
            <RateViolationsCard liveHealth={liveHealth} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <LatencyCard liveHealth={liveHealth} />
              <ProcessCard detailed={detailed} />
            </div>
          </div>
        </FoldSection>
      )}

      {/* Collapsible: Training & Quality */}
      {loaded && (
        <FoldSection heading="Training &amp; Quality">
          <div className="space-y-3">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {autoTrainStatus && <AutoTrainCard status={autoTrainStatus} />}
              <WorkflowCard onRefresh={fetchAll} />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {benchQuality && <QualityCard quality={benchQuality} stats={benchStats} />}
              <FeedbackCard
                dpoStatus={dpoStatus}
                visualStatus={visualStatus}
                dpoRunning={dpoRunning}
                onDpoRunningChange={setDpoRunning}
                onRefresh={fetchAll}
              />
            </div>
            {(trainingJobs.length > 0 || (executorStatus && executorStatus.initialized)) && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <TrainingHistory jobs={trainingJobs} />
                {executorStatus && <ExecutorPool status={executorStatus} onRefresh={fetchAll} />}
              </div>
            )}
            <InferencePoolCard onRefresh={fetchAll} />
          </div>
        </FoldSection>
      )}

      {/* Collapsible: System Info */}
      {loaded && (
        <FoldSection heading="System Info" open={false}>
          <div className="space-y-3">
            <KnowledgeCard knowledgeStats={knowledgeStats} adapterStatus={adapterStatus} loaded={loaded} />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <GpuCard gpu={detailed?.gpu as GPUInfo | undefined} />
              <DiskCard disk={disk ?? undefined} />
              <ServerInfoCard info={info ?? undefined} />
            </div>
            {detailed?.kv_sessions?.enabled && <KvCacheCard kvSessions={detailed.kv_sessions} />}
            {chartHistory.length > 1 && (
              <Card className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Real-time chart</span>
                  <button type="button" onClick={handleExportHistory} className="text-[10px] text-muted-foreground hover:text-primary transition-colors" aria-label="Export history">
                    Export
                  </button>
                </div>
                <CardContent className="p-0">
                  <div className="h-48" role="img" aria-label="CPU and memory usage chart over time">
                    <SystemChart data={chartHistory} />
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </FoldSection>
      )}

      {/* Collapsible: Service Output & Activity */}
      {loaded && (
        <FoldSection heading="Service Output" open={false}>
          <div className="space-y-3">
            <OutputCard compact />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="md:col-span-2">
                <ActivityTicker />
              </div>
              <Card className="p-4">
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Errors</span>
                <CardContent className="p-0 max-h-[200px] overflow-y-auto">
                  <ErrorList />
                </CardContent>
              </Card>
            </div>
          </div>
        </FoldSection>
      )}

    </PageContainer>
  )
})
