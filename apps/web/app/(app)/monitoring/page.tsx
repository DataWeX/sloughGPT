'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ProgressBar } from '@/components/ui'
import { StatCard, KpiGrid } from '@/components/ui/display'
import { systemController, type DetailedHealth, type SystemMetrics, type SystemInfo, type DiskUsage, type GPUInfo } from '@/lib/system-controller'
import { knowledgeController } from '@/lib/knowledge-controller'
import { useLocale } from '@/hooks/useLocale'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { PUBLIC_API_URL } from '@/lib/config'
import { formatUptime } from '@/lib/chat-utils'
import { GpuCard, DiskCard, ServerInfoCard } from '@/components/monitoring/SystemInfoCards'

export default function SystemHealthPage() {
  const { t } = useLocale()
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
  const [dpoStatus, setDpoStatus] = useState<{ status: string; last_run: string | null; accepted_count: number; rejected_count: number; result: any } | null>(null)
  const [dpoRunning, setDpoRunning] = useState(false)
  const [vlmStatus, setVlmStatus] = useState<{ vlm_loaded: boolean; training: { status: string } } | null>(null)
  const historyRef = useRef<Array<{ time: string; cpu: number; mem: number }>>([])
  const MAX_HISTORY = 30

  const fetchAll = useCallback(async (showRefreshing = false) => {
    if (showRefreshing) setRefreshing(true)
    setError(null)
    try {
      const [d, m, i, di, ks, as, bq, bs, ds, vs] = await Promise.all([
        systemController.getDetailedHealth(),
        systemController.getMetrics(),
        systemController.getInfo(),
        systemController.getDisk(),
        knowledgeController.stats(),
        knowledgeController.getAdapterStatus().catch(() => null),
        fetch(`${PUBLIC_API_URL}/benchmark/quality`).then(r => r.json()).catch(() => null),
        fetch(`${PUBLIC_API_URL}/benchmark/stats`).then(r => r.json()).catch(() => null),
        fetch(`${PUBLIC_API_URL}/vlm/dpo/status`).then(r => r.json()).catch(() => null),
        fetch(`${PUBLIC_API_URL}/vlm/status`).then(r => r.json()).catch(() => null),
      ])
      setDetailed(d)
      setMetrics(m)
      setInfo(i)
      setDisk(di)
      setKnowledgeStats(ks)
      setAdapterStatus(as)
      setBenchQuality(bq)
      setBenchStats(bs)
      setDpoStatus(ds)
      setVlmStatus(vs)
      // Append to rolling history
      if (m) {
        const h = historyRef.current
        h.push({ time: new Date().toLocaleTimeString(), cpu: m.cpu_percent, mem: m.memory_percent })
        if (h.length > MAX_HISTORY) h.shift()
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
          <Button variant="outline" size="sm" onClick={() => fetchAll(true)} disabled={refreshing || !loaded}>
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </Button>
        }
      />
      <div className="space-y-4">
        {/* Status + Model */}
        <Card>
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
                label="Inferences"
                value={!loaded ? '...' : detailed?.inference?.inference_count ?? 0}
              />
            </KpiGrid>
          </CardContent>
        </Card>

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
                label="Adapter"
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
            <CardHeader><CardTitle className="text-base">Model Quality</CardTitle></CardHeader>
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
                {benchStats && <span>Avg tokens: {benchStats.avg_tokens.toFixed(0)}</span>}
              </div>
            </CardContent>
          </Card>
        )}

        {/* DPO / VLM Training */}
        {dpoStatus || vlmStatus ? (
          <Card>
            <CardHeader><CardTitle className="text-base">Model Training (DPO + VLM)</CardTitle></CardHeader>
            <CardContent>
              <KpiGrid columns={4}>
                <StatCard
                  label="DPO Status"
                  value={dpoStatus ? dpoStatus.status : '...'}
                  icon={
                    <span className={`inline-block w-2 h-2 rounded-full ${!dpoStatus ? 'bg-warning' : dpoStatus.status === 'running' ? 'bg-warning' : dpoStatus.status === 'completed' ? 'bg-success' : dpoStatus.status === 'error' ? 'bg-destructive' : 'bg-muted-foreground/50'}`}
                    />
                  }
                />
                <StatCard
                  label="DPO Accepted"
                  value={dpoStatus ? dpoStatus.accepted_count.toString() : '...'}
                />
                <StatCard
                  label="DPO Rejected"
                  value={dpoStatus ? dpoStatus.rejected_count.toString() : '...'}
                />
                <StatCard
                  label="VLM Loaded"
                  value={vlmStatus ? (vlmStatus.vlm_loaded ? 'Yes' : 'No') : '...'}
                  icon={
                    <span className={`inline-block w-2 h-2 rounded-full ${!vlmStatus ? 'bg-warning' : vlmStatus.vlm_loaded ? 'bg-success' : 'bg-muted-foreground/50'}`}
                    />
                  }
                />
              </KpiGrid>
              {vlmStatus?.training && (
                <p className="text-xs text-muted-foreground mt-2">
                  VLM training: {vlmStatus.training.status}
                </p>
              )}
              {dpoStatus?.last_run && (
                <p className="text-xs text-muted-foreground mt-1">
                  Last DPO: {dpoStatus.last_run}
                </p>
              )}
              <div className="mt-3">
                <Button
                  size="sm"
                  disabled={dpoRunning || dpoStatus?.status === 'running'}
                  onClick={async () => {
                    setDpoRunning(true)
                    try {
                      await fetch(`${PUBLIC_API_URL}/vlm/dpo`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
                      await fetchAll()
                    } catch {}
                    setDpoRunning(false)
                  }}
                  aria-label="Run DPO training"
                >
                  {dpoRunning || dpoStatus?.status === 'running' ? 'Running...' : 'Run DPO'}
                </Button>
              </div>
            </CardContent>
          </Card>
        ) : null}

        {/* Real-time chart */}
        {historyRef.current.length > 1 && (
          <Card>
            <CardHeader><CardTitle className="text-base">Real‑time Metrics (last {MAX_HISTORY}s)</CardTitle></CardHeader>
            <CardContent>
              <div className="h-48" role="img" aria-label="CPU and memory usage chart over time">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={historyRef.current}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} width={30} />
                    <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6 }} />
                    <Line type="monotone" dataKey="cpu" stroke="var(--color-primary)" strokeWidth={1.5} dot={false} name="CPU %" />
                    <Line type="monotone" dataKey="mem" stroke="var(--color-warning)" strokeWidth={1.5} dot={false} name="Memory %" />
                  </LineChart>
                </ResponsiveContainer>
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
