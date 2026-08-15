'use client'
export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback, useRef } from 'react'
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
import { ProcessCard } from '@/components/monitoring/ProcessCard'
import { KvCacheCard } from '@/components/monitoring/KvCacheCard'
import { GpuCard, DiskCard, ServerInfoCard } from '@/components/monitoring/SystemInfoCards'
import { WorkflowCard } from '@/components/monitoring/WorkflowCard'
import { ActivityTicker, ErrorList } from '@/components/ActivityTicker'
import { OutputCard } from '@/components/OutputCard'

const POLL_INTERVAL_MS = 5000
const MAX_ALERT_HISTORY = 20

const SystemChart = dynamicNext(() => import('@/components/monitoring/SystemChart').then(m => m.SystemChart), {
  ssr: false,
  loading: () => <div className="h-40 w-full animate-pulse bg-muted rounded-lg" />,
})

const TrendChart = dynamicNext(() => import('@/components/monitoring/TrendChart').then(m => m.TrendChart), {
  ssr: false,
  loading: () => <div className="h-40 w-full animate-pulse bg-muted rounded-lg" />,
})

export default function SystemHealthPage() {
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
    status: string; total_responses: number; coherence_score: number; quality_score: number;
    repetition_rate: number; avg_length: number; empty_rate: number;
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
      setVisualStatus(vs as typeof visualStatus)
      setExecutorStatus(ex)
      setAutoTrainStatus(at)
      setTrainingJobs(Array.isArray(tj) ? tj : [])
      setLastUpdated(new Date().toLocaleTimeString())
    } catch (e: unknown) {
      setError(extractErrorMessage(e, 'Failed to load system health'))
    }
    if (showRefreshing) setRefreshing(false)
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

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
        const rate = ((count - prev.count) / elapsed) * 60
        setInferenceRate(Math.max(0, rate))
      }
    }
    prevInferenceRef.current = { count, time: now }
  }, [liveHealth])

  useEffect(() => {
    if (!loaded || !autoRefresh) return
    const poll = () => {
      if (document.hidden) return
      fetchAll()
    }
    const id = setInterval(poll, POLL_INTERVAL_MS)
    const handleVisibility = () => {
      if (!document.hidden && autoRefresh) fetchAll(true)
    }
    document.addEventListener('visibilitychange', handleVisibility)
    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', handleVisibility)
    }
  }, [loaded, fetchAll, autoRefresh])

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
        <span className="text-[11px] text-muted-foreground hidden sm:inline font-mono">Updated {lastUpdated}</span>
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
      {/* Loading skeletons */}
      {!loaded && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
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
              <LatencyCard chartHistory={chartHistory} />
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
                  <button onClick={handleExportHistory} className="text-[10px] text-muted-foreground hover:text-primary transition-colors" aria-label="Export history">
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

      {/* Collapsible: Server Output & Activity */}
      {loaded && (
        <FoldSection heading="Server Output" open={false}>
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

      {error && (
        <Card className="p-4 border-destructive/50">
          <CardContent className="p-0 py-2 text-sm text-destructive">{error}</CardContent>
        </Card>
      )}
    </PageContainer>
  )
}
