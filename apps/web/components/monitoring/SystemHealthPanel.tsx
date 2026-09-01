'use client'

import { memo, useMemo, useState } from 'react'
import { cn, Card, CardContent, Skeleton } from '@sloughgpt/strui'
import type { LiveHealthSnapshot, ConnectionStatus } from '@/hooks/useLiveStatus'
import type { DetailedHealth, SystemMetrics, DiskUsage, SystemInfo } from '@/lib/system-controller'
import { formatUptime } from '@/lib/chat-utils'
import { useTick } from '@/hooks/useTick'
import { timeAgo } from '@/lib/time-ago'

interface SystemHealthPanelProps {
  liveHealth: LiveHealthSnapshot | null
  detailed: DetailedHealth | null
  metrics: SystemMetrics | null
  disk: DiskUsage | null
  info: SystemInfo | null
  connectionStatus: ConnectionStatus
  loaded: boolean
  chartHistory: Array<{ time: string; cpu: number; mem: number; tokens?: number; latency?: number }>
  modelHealth?: { perplexity?: number; loss?: number; quality_score?: number; last_eval?: string; perplexity_trend?: Array<{ ts: string; value: number }>; loss_trend?: Array<{ ts: string; value: number }> } | null
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return `${n}`
}

function StatusDot({ active, className }: { active: boolean; className?: string }) {
  return (
    <span className={cn(
      'inline-block h-2 w-2 rounded-full shrink-0',
      active ? 'bg-success animate-pulse' : 'bg-muted-foreground/40',
      className
    )} />
  )
}

function ResourceBar({ value, threshold = 80, label }: { value: number; threshold?: number; label: string }) {
  const color = value > 90 ? 'bg-destructive' : value > threshold ? 'bg-warning' : 'bg-success'
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-muted-foreground">{label}</span>
        <span className="text-[11px] font-mono font-medium tabular-nums">{value.toFixed(1)}%</span>
      </div>
      <div className="relative h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={cn('absolute inset-y-0 left-0 rounded-full transition-all duration-500', color)}
          style={{ width: `${Math.min(100, value)}%` }}
        />
      </div>
    </div>
  )
}

function MiniSparkline({ data, className }: { data: number[]; className?: string }) {
  if (data.length < 2) return null
  const max = Math.max(...data, 1)
  const min = Math.min(...data, 0)
  const range = max - min || 1
  const w = 80
  const h = 20
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w
    const y = h - ((v - min) / range) * h
    return `${x},${y}`
  }).join(' ')
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className={cn('shrink-0', className)} aria-hidden="true">
      <polyline points={points} fill="none" stroke="rgb(var(--primary))" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.6" />
    </svg>
  )
}

function HealthRing({ score, status }: { score: number; status: string }) {
  const r = 28
  const c = 2 * Math.PI * r
  const offset = c - (score / 100) * c
  const color = score >= 80 ? 'rgb(var(--success, 34 197 94))' : score >= 50 ? 'rgb(var(--warning, 234 179 8))' : 'rgb(var(--destructive, 239 68 68))'
  return (
    <div className="relative flex items-center justify-center" style={{ width: 72, height: 72 }}>
      <svg viewBox="0 0 72 72" className="absolute inset-0 -rotate-90">
        <circle cx="36" cy="36" r={r} fill="none" stroke="rgb(var(--border))" strokeWidth="4" />
        <circle cx="36" cy="36" r={r} fill="none" stroke={color} strokeWidth="4" strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round" className="transition-all duration-700" />
      </svg>
      <div className="text-center z-10">
        <div className="text-sm font-mono font-semibold tabular-nums leading-none">{score}</div>
        <div className="text-[9px] text-muted-foreground uppercase tracking-wider mt-0.5">{status}</div>
      </div>
    </div>
  )
}

function KVPair({ label, value, mono = true, muted = false }: { label: string; value: React.ReactNode; mono?: boolean; muted?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-2 py-0.5">
      <span className="text-[11px] text-muted-foreground truncate">{label}</span>
      <span className={cn('text-[11px] text-right truncate', mono && 'font-mono', muted && 'text-muted-foreground')}>{value}</span>
    </div>
  )
}

function ExpandableStrip({ title, count, children }: { title: string; count?: number; children: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border-t border-border/50">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wider hover:bg-muted/30 transition-colors"
        aria-expanded={open}
      >
        <span>{title}</span>
        <span className="flex items-center gap-2">
          {count != null && count > 0 && (
            <span className="px-1.5 py-0.5 rounded-full bg-muted text-[9px] font-mono tabular-nums">{count}</span>
          )}
          <span className={cn('transition-transform duration-200', open && 'rotate-180')}>&#9660;</span>
        </span>
      </button>
      {open && <div className="px-4 pb-2">{children}</div>}
    </div>
  )
}

function LifecycleBadge({ lifecycle }: { lifecycle: DetailedHealth['lifecycle'] }) {
  if (!lifecycle) return null
  const phaseColor = lifecycle.phase === 'running' ? 'bg-success/10 text-success' :
    lifecycle.phase === 'draining' ? 'bg-warning/10 text-warning' :
    lifecycle.phase === 'starting' ? 'bg-primary/10 text-primary' :
    'bg-muted text-muted-foreground'
  return (
    <span className={cn('inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-medium', phaseColor)}>
      <span className={cn('h-1.5 w-1.5 rounded-full', lifecycle.is_running ? 'bg-success' : 'bg-muted-foreground/50')} />
      {lifecycle.phase}
      {lifecycle.profile && lifecycle.profile !== 'unknown' && <span className="text-[9px] opacity-60">/ {lifecycle.profile}</span>}
    </span>
  )
}

function ProcessGuardBadge({ pg }: { pg: DetailedHealth['process_guard'] }) {
  if (!pg) return null
  return (
    <span className={cn('inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-medium',
      pg.active ? 'bg-success/10 text-success' : pg.enabled ? 'bg-warning/10 text-warning' : 'bg-muted text-muted-foreground')}>
      <span className={cn('h-1.5 w-1.5 rounded-full', pg.active ? 'bg-success' : 'bg-muted-foreground/50')} />
      guard {pg.active ? 'active' : pg.enabled ? 'armed' : 'off'}
    </span>
  )
}

export const SystemHealthPanel = memo(function SystemHealthPanel({
  liveHealth, detailed, metrics, disk, info, connectionStatus, loaded, chartHistory, modelHealth,
}: SystemHealthPanelProps) {
  useTick()
  const h = liveHealth
  const sys = detailed?.system
  const gpu = detailed?.gpu
  const inf = detailed?.inference

  const cpu = h?.cpu_percent ?? metrics?.cpu_percent ?? sys?.cpu_percent ?? null
  const memPct = h?.memory_percent ?? metrics?.memory_percent ?? sys?.memory_percent ?? null
  const memUsed = metrics?.memory_used_gb ?? (sys?.rss_mb != null ? sys.rss_mb / 1024 : null)
  const memTotal = metrics?.memory_total_gb ?? null
  const memAvail = sys?.memory_available_mb != null ? sys.memory_available_mb / 1024 : null

  const cpuHistory = useMemo(() => chartHistory.map(p => p.cpu), [chartHistory])
  const memHistory = useMemo(() => chartHistory.map(p => p.mem), [chartHistory])

  const healthScore = h?.health_score ?? detailed?.health_score?.score ?? 0
  const healthStatus = h?.health_status ?? detailed?.health_score?.status ?? 'unknown'
  const modelLoaded = h?.model_loaded ?? detailed?.model_loaded ?? false
  const modelLoading = h?.model_loading ?? detailed?.model_loading ?? false
  const modelType = h?.model_type ?? detailed?.model_type ?? null
  const params = h?.num_parameters ?? detailed?.num_parameters ?? null
  const soul = h?.soul ?? detailed?.soul ?? null
  const device = h?.device ?? detailed?.device ?? null
  const requestCount = h?.request_count ?? detailed?.request_count ?? 0
  const pathLatencies = h?.path_latencies ?? []
  const recentErrors = h?.recent_errors ?? []
  const modelMetrics = h?.model_metrics ?? []
  const rateViolations = h?.rate_violations ?? []
  const trainingPool = h?.training_pool ?? detailed?.training_pool ?? null
  const kvSessions = detailed?.kv_sessions
  const quantization = detailed?.quantization
  const lifecycle = detailed?.lifecycle
  const resourceAlloc = detailed?.resource_allocation
  const registry = detailed?.registry
  const memPressure = detailed?.memory_pressure
  const processGuard = detailed?.process_guard
  const mpsMonitor = detailed?.mps_monitor
  const idleManager = detailed?.idle
  const gpuMemoryHint = gpu?.memory_hint ? (() => { try { return JSON.parse(gpu.memory_hint) as Record<string, unknown> } catch { return null } })() : null

  const connColor = connectionStatus === 'connected' ? 'text-success' : connectionStatus === 'connecting' ? 'text-warning' : 'text-destructive'
  const connLabel = connectionStatus === 'connected' ? 'Connected' : connectionStatus === 'connecting' ? 'Reconnecting' : 'Offline'

  if (!loaded) {
    return (
      <Card className="p-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="space-y-2">
              <Skeleton className="h-3 w-16" />
              <Skeleton className="h-5 w-20" />
            </div>
          ))}
        </div>
      </Card>
    )
  }

  return (
    <Card className="p-0 overflow-hidden" role="region" aria-label="System health dashboard">
      {/* Top bar: Health score + connection + uptime + badges */}
      <div className="flex items-center gap-4 px-4 py-3 border-b border-border/50">
        <HealthRing score={healthScore} status={healthStatus} />
        <div className="flex-1 min-w-0 space-y-1">
          <div className="flex items-center gap-2 flex-wrap">
            <StatusDot active={connectionStatus === 'connected'} />
            <span className={cn('text-xs font-medium', connColor)}>{connLabel}</span>
            {h?.is_inferencing && (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-primary/10 text-primary text-[10px] font-medium">
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
                inferencing
              </span>
            )}
            <LifecycleBadge lifecycle={lifecycle} />
            <ProcessGuardBadge pg={processGuard} />
            {idleManager?.enabled && (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-muted/50 text-muted-foreground text-[10px] font-medium">
                idle {idleManager.idle_seconds != null ? `${Math.round(idleManager.idle_seconds)}s` : ''}
              </span>
            )}
            {lifecycle?.error && (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-destructive/10 text-destructive text-[10px] font-medium" title={lifecycle.error}>
                lifecycle error
              </span>
            )}
          </div>
          <div className="text-xs text-muted-foreground">
            Uptime <span className="font-mono tabular-nums">{formatUptime(h?.uptime_seconds ?? detailed?.uptime_seconds ?? 0)}</span>
            {detailed?.timestamp && (
              <span className="ml-2 text-[10px] opacity-60">snapshot {timeAgo(new Date(detailed.timestamp).getTime() / 1000)}</span>
            )}
          </div>
          {h?.health_summary && <p className="text-[11px] text-muted-foreground/70 line-clamp-1">{h.health_summary}</p>}
        </div>
        {/* Mini sparklines */}
        {cpuHistory.length > 1 && (
          <div className="hidden md:flex items-center gap-3 shrink-0">
            <div className="text-right">
              <div className="text-[10px] text-muted-foreground">CPU</div>
              <MiniSparkline data={cpuHistory} className="w-20 h-5" />
            </div>
            <div className="text-right">
              <div className="text-[10px] text-muted-foreground">MEM</div>
              <MiniSparkline data={memHistory} className="w-20 h-5" />
            </div>
          </div>
        )}
      </div>

      {/* Dense 4-column grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 divide-y md:divide-y-0 md:divide-x divide-border/50">
        {/* Col 1: Resources */}
        <div className="p-3 space-y-3">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Resources</span>
            <StatusDot active={cpu != null && cpu < 80} className="!h-1.5 !w-1.5" />
          </div>
          <div className="space-y-2">
            {cpu != null && <ResourceBar value={cpu} label="CPU" />}
            {memPct != null && <ResourceBar value={memPct} label="Memory" />}
          </div>
          <div className="space-y-0.5">
            {memUsed != null && memTotal != null && (
              <KVPair label="Used / Total" value={`${memUsed.toFixed(1)} / ${memTotal.toFixed(0)} GB`} />
            )}
            {memAvail != null && <KVPair label="Available" value={`${memAvail.toFixed(1)} GB`} />}
            {disk && (
              <>
                <KVPair label="Disk" value={`${disk.used_gb.toFixed(0)} / ${disk.total_gb.toFixed(0)} GB (${disk.percent}%)`} />
                <KVPair label="Disk free" value={`${disk.free_gb.toFixed(1)} GB`} />
              </>
            )}
            {sys?.rss_mb != null && <KVPair label="RSS" value={`${sys.rss_mb.toFixed(0)} MB`} />}
            {memPressure && (
              <>
                <div className="border-t border-border/30 my-1" />
                {memPressure.current_mb != null && <KVPair label="Current RSS" value={`${memPressure.current_mb.toFixed(0)} MB`} muted />}
                {memPressure.peak_mb != null && <KVPair label="Peak RSS" value={`${memPressure.peak_mb.toFixed(0)} MB`} muted />}
                {memPressure.pressure_level && <KVPair label="Pressure" value={
                  <span className={cn(memPressure.pressure_level === 'high' ? 'text-destructive' : memPressure.pressure_level === 'moderate' ? 'text-warning' : '')}>
                    {memPressure.pressure_level}
                  </span>
                } muted />}
                {memPressure.tracked_count != null && <KVPair label="Tracked allocs" value={memPressure.tracked_count} muted />}
              </>
            )}
            {mpsMonitor && (
              <>
                <div className="border-t border-border/30 my-1" />
                <KVPair label="MPS usage" value={`${(mpsMonitor.usage * 100).toFixed(1)}%`} muted />
                <KVPair label="MPS locked" value={mpsMonitor.locked_to_cpu ? 'CPU' : 'GPU'} muted />
              </>
            )}
          </div>
        </div>

        {/* Col 2: Model */}
        <div className="p-3 space-y-3">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Model</span>
            <StatusDot active={modelLoaded && !modelLoading} className="!h-1.5 !w-1.5" />
          </div>
          <div className="space-y-0.5">
            <KVPair label="Status" value={
              modelLoading ? <span className="text-warning">Loading...</span> :
              modelLoaded ? <span className="text-success">Loaded</span> :
              <span className="text-muted-foreground">Not loaded</span>
            } />
            {modelType && <KVPair label="Model" value={modelType} />}
            {params != null && params > 0 && (
              <KVPair label="Parameters" value={params >= 1e9 ? `${(params / 1e9).toFixed(1)}B` : `${Math.round(params / 1e6)}M`} />
            )}
            {device && <KVPair label="Device" value={device} />}
            {soul && <KVPair label="Soul" value={soul} />}
            {gpu && (
              <>
                <KVPair label="GPU" value={
                  <span className={cn(gpu.backend && gpu.backend !== 'none' ? 'text-success' : 'text-muted-foreground')}>
                    {gpu.backend} · {gpu.tier}
                  </span>
                } />
                {gpu.device_type && <KVPair label="GPU device" value={gpu.device_type} muted />}
                {gpu.vram_gb > 0 && <KVPair label="VRAM" value={`${gpu.vram_gb} GB`} />}
                {gpuMemoryHint && Object.entries(gpuMemoryHint).filter(([k]) => !['tier'].includes(k)).slice(0, 3).map(([k, v]) => (
                  <KVPair key={k} label={k.replace(/_/g, ' ')} value={typeof v === 'boolean' ? (v ? 'Yes' : 'No') : String(v)} muted />
                ))}
              </>
            )}
            {quantization != null && (
              <KVPair label="Quantization" value={
                typeof quantization === 'string' ? quantization :
                typeof quantization === 'object' && quantization !== null && 'bits' in quantization
                  ? `${(quantization as { bits: number }).bits}-bit`
                  : JSON.stringify(quantization)
              } muted />
            )}
            {registry && (
              <>
                <div className="border-t border-border/30 my-1" />
                <KVPair label="Registry" value={
                  <span className={cn(registry.healthy ? 'text-success' : 'text-destructive')}>
                    {registry.healthy ? 'Healthy' : 'Degraded'}
                  </span>
                } muted />
                {registry.default_model && <KVPair label="Default" value={registry.default_model} muted />}
                {registry.models && registry.models.length > 0 && <KVPair label="Models" value={registry.models.length} muted />}
              </>
            )}
          </div>
        </div>

        {/* Col 3: Inference */}
        <div className="p-3 space-y-3">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Inference</span>
            <StatusDot active={(h?.tokens_per_sec ?? 0) > 0} className="!h-1.5 !w-1.5" />
          </div>
          <div className="space-y-0.5">
            <KVPair label="Requests" value={String(requestCount)} />
            <KVPair label="Responses" value={String(h?.inference_count ?? inf?.inference_count ?? 0)} />
            <KVPair label="Tokens/sec" value={
              h?.tokens_per_sec != null ? h.tokens_per_sec.toFixed(1) : '—'
            } />
            <KVPair label="Total tokens" value={formatTokens(h?.total_tokens ?? 0)} />
            {inf?.total_generated != null && inf.total_generated > 0 && (
              <KVPair label="Generated" value={formatTokens(inf.total_generated)} />
            )}
            <KVPair label="Avg tokens/req" value={
              (h?.avg_tokens_per_request ?? 0) > 0 ? (h?.avg_tokens_per_request ?? 0).toFixed(0) : '—'
            } />
            <KVPair label="Avg latency" value={
              (h?.avg_latency_ms ?? 0) > 0 ? `${(h?.avg_latency_ms ?? 0).toFixed(0)}ms` : '—'
            } />
            <KVPair label="P95 latency" value={
              (h?.p95_latency_ms ?? 0) > 0 ? `${(h?.p95_latency_ms ?? 0).toFixed(0)}ms` : '—'
            } />
            <KVPair label="Requests/min" value={
              (h?.requests_per_minute ?? 0) > 0 ? (h?.requests_per_minute ?? 0).toFixed(1) : '—'
            } />
            <KVPair label="Errors" value={
              <span className={cn((h?.error_count ?? 0) > 0 ? 'text-destructive' : '')}>
                {h?.error_count ?? 0}
              </span>
            } />
            {lifecycle?.in_flight != null && lifecycle.in_flight > 0 && (
              <KVPair label="In-flight" value={lifecycle.in_flight} />
            )}
          </div>
        </div>

        {/* Col 4: Process & System */}
        <div className="p-3 space-y-3">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Process</span>
            <StatusDot active={true} className="!h-1.5 !w-1.5" />
          </div>
          <div className="space-y-0.5">
            {sys?.process_cpu_percent != null && <KVPair label="Proc CPU" value={`${sys.process_cpu_percent}%`} />}
            {sys?.process_memory_percent != null && <KVPair label="Proc Memory" value={`${sys.process_memory_percent}%`} />}
            {sys?.threads != null && <KVPair label="Threads" value={sys.threads} />}
            {sys?.open_files != null && <KVPair label="Open files" value={sys.open_files} />}
            {sys?.gc_gen0 != null && <KVPair label="GC Gen0/1/2" value={`${sys.gc_gen0} / ${sys.gc_gen1 ?? 0} / ${sys.gc_gen2 ?? 0}`} />}
            {trainingPool && (
              <>
                <div className="border-t border-border/30 my-1" />
                <KVPair label="Train pool" value={`${trainingPool.active_jobs} / ${trainingPool.max_workers}`} muted />
                <KVPair label="Tracked jobs" value={trainingPool.total_tracked} muted />
              </>
            )}
            {kvSessions?.enabled && (
              <>
                <div className="border-t border-border/30 my-1" />
                <KVPair label="KV sessions" value={`${kvSessions.active_sessions ?? 0} / ${kvSessions.max_sessions ?? 0}`} muted />
                <KVPair label="Cached tokens" value={formatTokens(kvSessions.cached_tokens ?? 0)} muted />
                {kvSessions.ttl_seconds != null && <KVPair label="KV TTL" value={`${kvSessions.ttl_seconds}s`} muted />}
                {kvSessions.oldest_session_age != null && <KVPair label="Oldest session" value={`${kvSessions.oldest_session_age.toFixed(0)}s`} muted />}
              </>
            )}
            {processGuard && (
              <>
                <div className="border-t border-border/30 my-1" />
                <KVPair label="Guard" value={
                  <span className={cn(processGuard.active ? 'text-success' : processGuard.enabled ? 'text-warning' : 'text-muted-foreground')}>
                    {processGuard.active ? 'Active' : processGuard.enabled ? 'Armed' : 'Off'}
                  </span>
                } muted />
                {processGuard.health && (
                  <>
                    <KVPair label="Guard alive" value={processGuard.health.alive ? 'Yes' : 'No'} muted />
                    {processGuard.health.memory_mb != null && <KVPair label="Guard mem" value={`${processGuard.health.memory_mb.toFixed(0)} MB`} muted />}
                    {processGuard.health.restarts != null && processGuard.health.restarts > 0 && <KVPair label="Restarts" value={processGuard.health.restarts} muted />}
                  </>
                )}
              </>
            )}
            {info && (
              <>
                <div className="border-t border-border/30 my-1" />
                <KVPair label="Platform" value={`${info.platform} ${info.platform_release}`} muted />
                <KVPair label="Arch" value={info.architecture} muted />
                <KVPair label="Cores" value={info.cpu_count} muted />
              </>
            )}
          </div>
        </div>
      </div>

      {/* Diagnostics strip */}
      {h && h.diagnoses.length > 0 && (
        <div className="px-4 py-2 border-t border-border/50 bg-muted/30">
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {h.diagnoses.slice(0, 6).map((d, i) => (
              <div key={`${d.check}-${i}`} className="flex items-center gap-1.5 text-[10px]">
                <span className={cn('h-1.5 w-1.5 rounded-full shrink-0', d.severity === 'critical' ? 'bg-destructive' : d.severity === 'warn' ? 'bg-warning' : d.severity === 'info' ? 'bg-primary' : 'bg-success')} />
                <span className="font-medium capitalize">{d.check}</span>
                <span className="text-muted-foreground">{d.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Expandable detail sections */}
      {pathLatencies.length > 0 && (
        <ExpandableStrip title="Endpoint Latency" count={pathLatencies.length}>
          <div className="space-y-1">
            {pathLatencies.map((p) => (
              <div key={p.path} className="flex items-center gap-2 text-[10px] py-0.5">
                <span className="font-mono truncate min-w-0 flex-1">{p.path}</span>
                <span className="shrink-0 px-1.5 py-0.5 rounded bg-muted font-mono tabular-nums">x{p.count}</span>
                <span className="shrink-0 px-1.5 py-0.5 rounded bg-primary/10 text-primary font-mono tabular-nums">{p.avg_ms.toFixed(1)}ms</span>
                <span className="shrink-0 px-1.5 py-0.5 rounded bg-warning/10 text-warning font-mono tabular-nums">p95 {p.p95_ms.toFixed(1)}ms</span>
              </div>
            ))}
          </div>
        </ExpandableStrip>
      )}

      {modelMetrics.length > 0 && (
        <ExpandableStrip title="Model Metrics" count={modelMetrics.length}>
          <div className="space-y-1">
            {modelMetrics.map((m) => (
              <div key={m.model} className="flex items-center gap-2 text-[10px] py-0.5">
                <span className="font-mono truncate min-w-0 flex-1">{m.model}</span>
                <span className="shrink-0 px-1.5 py-0.5 rounded bg-muted font-mono tabular-nums">x{m.count}</span>
                <span className="shrink-0 px-1.5 py-0.5 rounded bg-primary/10 text-primary font-mono tabular-nums">{formatTokens(m.total_tokens)} tok</span>
                <span className="shrink-0 px-1.5 py-0.5 rounded bg-success/10 text-success font-mono tabular-nums">{m.tokens_per_sec.toFixed(1)} t/s</span>
                <span className="shrink-0 px-1.5 py-0.5 rounded bg-muted font-mono tabular-nums">avg {m.avg_tokens.toFixed(0)}</span>
              </div>
            ))}
          </div>
        </ExpandableStrip>
      )}

      {recentErrors.length > 0 && (
        <ExpandableStrip title="Recent Errors" count={recentErrors.length}>
          <div className="space-y-1 max-h-[160px] overflow-y-auto">
            {recentErrors.slice(0, 8).map((e, i) => (
              <div key={`${e.ts}-${i}`} className="flex items-start gap-2 text-[10px] py-0.5 border border-destructive/20 rounded px-2 py-1">
                <span className="shrink-0 px-1.5 py-0.5 rounded bg-destructive/10 text-destructive font-mono font-medium">{e.status}</span>
                <span className="font-mono truncate flex-1">{e.method} {e.path}</span>
                <span className="text-muted-foreground truncate max-w-[140px]" title={e.message}>
                  {e.error_type && <span className="text-destructive/80">{e.error_type}: </span>}
                  {e.message}
                </span>
                <span className="shrink-0 text-muted-foreground/60 font-mono">{timeAgo(e.ts)}</span>
              </div>
            ))}
          </div>
        </ExpandableStrip>
      )}

      {rateViolations.length > 0 && (
        <ExpandableStrip title="Rate Violations" count={rateViolations.length}>
          <div className="space-y-1">
            {rateViolations.map((v, i) => (
              <div key={`${v.path}-${i}`} className="flex items-center gap-2 text-[10px] py-0.5">
                <span className="font-mono truncate min-w-0 flex-1">{v.path}</span>
                <span className="shrink-0 px-1.5 py-0.5 rounded bg-destructive/10 text-destructive font-mono tabular-nums">{v.count} hits</span>
                <span className="shrink-0 px-1.5 py-0.5 rounded bg-muted font-mono tabular-nums">limit {v.limit}</span>
                <span className="shrink-0 text-muted-foreground/60 font-mono">{timeAgo(v.ts)}</span>
              </div>
            ))}
          </div>
        </ExpandableStrip>
      )}

      {resourceAlloc && Object.keys(resourceAlloc).length > 0 && (
        <ExpandableStrip title="Resource Allocation">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-x-4 gap-y-0.5">
            {resourceAlloc.mode && <KVPair label="Mode" value={resourceAlloc.mode} />}
            {resourceAlloc.compute_threads != null && <KVPair label="Compute threads" value={resourceAlloc.compute_threads} />}
            {resourceAlloc.io_threads != null && <KVPair label="IO threads" value={resourceAlloc.io_threads} />}
            {resourceAlloc.omp_num_threads != null && <KVPair label="OMP threads" value={resourceAlloc.omp_num_threads} />}
            {resourceAlloc.mkl_num_threads != null && <KVPair label="MKL threads" value={resourceAlloc.mkl_num_threads} />}
            {resourceAlloc.openblas_num_threads != null && <KVPair label="OpenBLAS threads" value={resourceAlloc.openblas_num_threads} />}
            {resourceAlloc.numexpr_num_threads != null && <KVPair label="NumExpr threads" value={resourceAlloc.numexpr_num_threads} />}
            {resourceAlloc.inference_pool_size != null && <KVPair label="Inference pool" value={resourceAlloc.inference_pool_size} />}
            {resourceAlloc.train_pool_size != null && <KVPair label="Train pool" value={resourceAlloc.train_pool_size} />}
            {resourceAlloc.task_queue_workers != null && <KVPair label="Task queue" value={resourceAlloc.task_queue_workers} />}
            {resourceAlloc.dataloader_workers != null && <KVPair label="Dataloader" value={resourceAlloc.dataloader_workers} />}
            {resourceAlloc.concurrent_reads != null && <KVPair label="Concurrent reads" value={resourceAlloc.concurrent_reads} />}
            {resourceAlloc.concurrent_writes != null && <KVPair label="Concurrent writes" value={resourceAlloc.concurrent_writes} />}
            {resourceAlloc.process_guard_concurrent != null && <KVPair label="Guard concurrent" value={resourceAlloc.process_guard_concurrent} />}
          </div>
        </ExpandableStrip>
      )}

      {modelHealth && (modelHealth.perplexity != null || modelHealth.loss != null || modelHealth.quality_score != null) && (
        <ExpandableStrip title="Model Health">
          <div className="space-y-1">
            {modelHealth.perplexity != null && <KVPair label="Perplexity" value={modelHealth.perplexity.toFixed(3)} />}
            {modelHealth.loss != null && <KVPair label="Loss" value={modelHealth.loss.toFixed(4)} />}
            {modelHealth.quality_score != null && <KVPair label="Quality" value={
              <span className={cn(modelHealth.quality_score >= 0.7 ? 'text-success' : modelHealth.quality_score >= 0.4 ? 'text-warning' : 'text-destructive')}>
                {(modelHealth.quality_score * 100).toFixed(1)}%
              </span>
            } />}
            {modelHealth.last_eval && <KVPair label="Last eval" value={timeAgo(modelHealth.last_eval)} muted />}
            {modelHealth.perplexity_trend && modelHealth.perplexity_trend.length > 1 && (
              <div className="mt-1">
                <span className="text-[10px] text-muted-foreground">Perplexity trend</span>
                <MiniSparkline data={modelHealth.perplexity_trend.map(p => p.value)} className="w-full h-6 mt-0.5" />
              </div>
            )}
            {modelHealth.loss_trend && modelHealth.loss_trend.length > 1 && (
              <div className="mt-1">
                <span className="text-[10px] text-muted-foreground">Loss trend</span>
                <MiniSparkline data={modelHealth.loss_trend.map(p => p.value)} className="w-full h-6 mt-0.5" />
              </div>
            )}
          </div>
        </ExpandableStrip>
      )}

      {/* Model events strip */}
      {h && h.model_events.length > 0 && (
        <div className="px-4 py-2 border-t border-border/50">
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {h.model_events.slice(0, 4).map((e, i) => (
              <div key={`${e.ts}-${i}`} className="flex items-center gap-1.5 text-[10px]">
                <span className={cn('h-1.5 w-1.5 rounded-full shrink-0', e.type === 'load' ? 'bg-success' : e.type === 'unload' ? 'bg-warning' : 'bg-primary')} />
                <span className="font-medium capitalize">{e.type}</span>
                <span className="text-muted-foreground truncate max-w-[120px]">{e.model}</span>
                <span className="text-muted-foreground/60">{e.detail}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  )
})
