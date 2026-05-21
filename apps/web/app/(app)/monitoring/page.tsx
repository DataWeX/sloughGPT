'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ProgressBar } from '@/components/ui'
import { StatCard, KpiGrid } from '@/components/ui/display'
import { systemController, type DetailedHealth, type SystemMetrics, type SystemInfo, type DiskUsage, type GPUInfo } from '@/lib/system-controller'
import { useLocale } from '@/hooks/useLocale'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400)
  const h = Math.floor((seconds % 86400) / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (d > 0) return `${d}d ${h}h ${m}m`
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

function GpuCard({ gpu }: { gpu?: GPUInfo }) {
  if (!gpu) return null
  return (
    <Card>
      <CardHeader><CardTitle className="text-base">GPU</CardTitle></CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div className="flex justify-between"><span className="text-muted-foreground">Backend</span><span>{gpu.backend}</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">Device</span><span>{gpu.device_type}</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">VRAM</span><span>{gpu.vram_gb} GB</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">Tier</span><span>{gpu.tier}</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">Memory hint</span><span className="text-xs max-w-[180px] text-right">{gpu.memory_hint}</span></div>
      </CardContent>
    </Card>
  )
}

function DiskCard({ disk }: { disk?: DiskUsage }) {
  if (!disk) return null
  const pct = Math.round(disk.percent)
  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Disk</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{disk.used_gb.toFixed(1)} GB used</span>
          <span>{disk.total_gb.toFixed(1)} GB total</span>
        </div>
        <ProgressBar value={pct} max={100} />
        <p className="text-xs text-muted-foreground">{disk.free_gb.toFixed(1)} GB free</p>
      </CardContent>
    </Card>
  )
}

function ServerInfoCard({ info }: { info?: SystemInfo }) {
  if (!info) return null
  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Server</CardTitle></CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div className="flex justify-between"><span className="text-muted-foreground">Platform</span><span>{info.platform} {info.platform_release}</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">Architecture</span><span>{info.architecture}</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">CPU cores</span><span>{info.cpu_count}</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">Processor</span><span className="text-xs max-w-[200px] text-right truncate" title={info.processor}>{info.processor || '—'}</span></div>
      </CardContent>
    </Card>
  )
}

export default function SystemHealthPage() {
  const { t } = useLocale()
  const [detailed, setDetailed] = useState<DetailedHealth | null>(null)
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [disk, setDisk] = useState<DiskUsage | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const historyRef = useRef<Array<{ time: string; cpu: number; mem: number }>>([])
  const MAX_HISTORY = 30

  const fetchAll = useCallback(async (showRefreshing = false) => {
    if (showRefreshing) setRefreshing(true)
    setError(null)
    try {
      const [d, m, i, di] = await Promise.all([
        systemController.getDetailedHealth(),
        systemController.getMetrics(),
        systemController.getInfo(),
        systemController.getDisk(),
      ])
      setDetailed(d)
      setMetrics(m)
      setInfo(i)
      setDisk(di)
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

        {/* Real-time chart */}
        {historyRef.current.length > 1 && (
          <Card>
            <CardHeader><CardTitle className="text-base">Real‑time Metrics (last {MAX_HISTORY}s)</CardTitle></CardHeader>
            <CardContent>
              <div className="h-48">
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
