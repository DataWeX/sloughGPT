'use client'

import { useState, useEffect, useRef } from 'react'
import { useErrorStore } from '@/lib/error-store'
import { PUBLIC_API_URL } from '@/lib/config'
import { cn } from '@/lib/cn'

const API = PUBLIC_API_URL || 'http://localhost:8000'

interface HealthScore {
  score: number
  status: string
  error_rate_score: number
  latency_score: number
  throughput_score: number
  uptime_score: number
}

interface ModelMetric {
  model: string
  count: number
  total_tokens: number
  tokens_per_sec: number
  avg_tokens: number
}

interface BackendDebug {
  model_loaded: boolean
  model_type: string | null
  soul: string | null
  uptime_seconds: number
  request_count: number
  error_count: number
  inference_count: number
  total_tokens: number
  tokens_per_sec: number
  avg_tokens_per_request: number
  avg_latency_ms: number
  requests_per_minute: number
  health_score: HealthScore
  model_metrics: ModelMetric[]
  model_events: Array<{ type: string; model: string; detail: string; ts: number }>
  health_history: Array<{ score: number; status: string; ts: number }>
  memory_history: Array<{ rss_mb: number; virtual_mb: number; system_percent: number; ts: number }>
  rate_violations: Array<{ path: string; count: number; limit: number; ts: number }>
  path_latencies: Array<{ path: string; avg_ms: number; count: number; p95_ms: number }>
  recent_errors: Array<{ path: string; method: string; status: number; message: string; error_type: string; ts: number }>
  recent_requests: Array<{ path: string; method: string; status: number; elapsed_ms: number }>
  cpu_percent: number | null
  memory_percent: number | null
  gpu_backend: string | null
}

interface DebugOverlayProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

function scoreColor(score: number): string {
  if (score >= 80) return 'text-green-400'
  if (score >= 50) return 'text-yellow-400'
  return 'text-red-400'
}

function Sparkline({ data, width = 80, height = 16 }: { data: number[]; width?: number; height?: number }) {
  if (data.length < 2) return null
  const min = Math.min(...data)
  const max = Math.max(...data) || 100
  const range = max - min || 1
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width
    const y = height - ((v - min) / range) * height
    return `${x},${y}`
  }).join(' ')
  return (
    <svg width={width} height={height} className="inline-block">
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        className={scoreColor(data[data.length - 1])}
      />
    </svg>
  )
}

function scoreBg(score: number): string {
  if (score >= 80) return 'bg-green-400/10 border-green-400/20'
  if (score >= 50) return 'bg-yellow-400/10 border-yellow-400/20'
  return 'bg-red-400/10 border-red-400/20'
}

export function DebugOverlay({ open, onOpenChange }: DebugOverlayProps) {
  const errors = useErrorStore(s => s.errors)
  const [live, setLive] = useState<Record<string, string>>({})
  const [healthScore, setHealthScore] = useState<HealthScore | null>(null)
  const [modelMetrics, setModelMetrics] = useState<ModelMetric[]>([])
  const [modelEvents, setModelEvents] = useState<BackendDebug['model_events']>([])
  const [healthHistory, setHealthHistory] = useState<BackendDebug['health_history']>([])
  const [memoryHistory, setMemoryHistory] = useState<BackendDebug['memory_history']>([])
  const [rateViolations, setRateViolations] = useState<BackendDebug['rate_violations']>([])
  const [pathLats, setPathLats] = useState<BackendDebug['path_latencies']>([])
  const [recentErrs, setRecentErrs] = useState<BackendDebug['recent_errors']>([])
  const [recentReqs, setRecentReqs] = useState<BackendDebug['recent_requests']>([])
  const timerRef = useRef<ReturnType<typeof setInterval>>()

  useEffect(() => {
    if (!open) {
      if (timerRef.current) clearInterval(timerRef.current)
      return
    }

    const refresh = async () => {
      let backend: BackendDebug | null = null
      try {
        const r = await fetch(`${API}/health/debug`, { signal: AbortSignal.timeout(3000) })
        if (r.ok) backend = await r.json()
      } catch {}

      const frontendErrCount = useErrorStore.getState().errors.length

      setLive({
        model: backend?.model_type || (backend?.model_loaded ? 'loaded' : '—'),
        soul: backend?.soul || '—',
        uptime: backend ? `${backend.uptime_seconds.toFixed(0)}s` : '—',
        requests: backend ? String(backend.request_count) : '—',
        'req/min': backend ? backend.requests_per_minute.toFixed(0) : '—',
        'srverrors': backend ? String(backend.error_count) : '—',
        inferences: backend ? String(backend.inference_count) : '—',
        'tok/sec': backend ? `${backend.tokens_per_sec.toFixed(1)}` : '—',
        'avg tok': backend ? `${backend.avg_tokens_per_request.toFixed(0)}` : '—',
        latency: backend ? `${backend.avg_latency_ms.toFixed(0)}ms` : '—',
        cpu: backend?.cpu_percent != null ? `${backend.cpu_percent.toFixed(0)}%` : '—',
        mem: backend?.memory_percent != null ? `${backend.memory_percent.toFixed(0)}%` : '—',
        gpu: backend?.gpu_backend || '—',
        'fe errors': String(frontendErrCount),
      })
      setHealthScore(backend?.health_score || null)
      setModelMetrics(backend?.model_metrics || [])
      setModelEvents(backend?.model_events || [])
      setHealthHistory(backend?.health_history || [])
      setMemoryHistory(backend?.memory_history || [])
      setRateViolations(backend?.rate_violations || [])
      setPathLats(backend?.path_latencies || [])
      setRecentErrs(backend?.recent_errors || [])
      setRecentReqs(backend?.recent_requests || [])
    }

    refresh()
    timerRef.current = setInterval(refresh, 3000)
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [open])

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'D') {
        e.preventDefault()
        onOpenChange(!open)
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [open, onOpenChange])

  const lastError = errors[errors.length - 1]

  if (!open) return null

  return (
    <div className="fixed bottom-16 right-4 z-[300] w-80 rounded-lg border border-border/60 bg-background/95 backdrop-blur-md shadow-2xl text-xs font-mono">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/30">
        <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Debug</span>
        <div className="flex items-center gap-2">
          <span className="text-[8px] text-muted-foreground/50">^⇧D</span>
          <button
            onClick={() => onOpenChange(false)}
            className="p-0.5 rounded hover:bg-muted/60 transition-colors"
            aria-label="Close debug overlay"
          >
            <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
      <div className="max-h-72 overflow-y-auto p-2 space-y-0.5">
        {healthScore && (
          <div className={cn("rounded-md border p-2 mb-1", scoreBg(healthScore.score))}>
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-semibold text-muted-foreground">HEALTH</span>
              <div className="flex items-center gap-2">
                {healthHistory.length >= 2 && (
                  <Sparkline data={healthHistory.map(h => h.score)} width={60} height={12} />
                )}
                <span className={cn("text-lg font-bold tabular-nums", scoreColor(healthScore.score))}>
                  {healthScore.score}
                </span>
              </div>
            </div>
            <div className="grid grid-cols-4 gap-1 mt-1">
              {[
                { label: 'err', score: healthScore.error_rate_score },
                { label: 'lat', score: healthScore.latency_score },
                { label: 'tps', score: healthScore.throughput_score },
                { label: 'up', score: healthScore.uptime_score },
              ].map(({ label, score: s }) => (
                <div key={label} className="text-center">
                  <div className="text-[8px] text-muted-foreground/60">{label}</div>
                  <div className={cn("text-[10px] tabular-nums", scoreColor(s))}>{s}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        {Object.entries(live).map(([key, val]) => (
          <div key={key} className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground/70 shrink-0">{key}</span>
            <span className={cn(
              "text-right truncate max-w-[180px]",
              key === 'srverrors' && val !== '0' && val !== '—' ? 'text-destructive' : '',
            )} title={val}>{val || <span className="opacity-30">—</span>}</span>
          </div>
        ))}
        {modelMetrics.length > 0 && (
          <div className="mt-2 pt-2 border-t border-border/30">
            <div className="text-muted-foreground/50 text-[9px] mb-1">Models</div>
            <div className="space-y-0.5">
              {modelMetrics.map((m, i) => (
                <div key={i} className="flex items-center justify-between gap-2 text-[10px]">
                  <span className="text-muted-foreground/60 truncate">{m.model}</span>
                  <span className="text-muted-foreground/40 tabular-nums shrink-0">{m.tokens_per_sec.toFixed(1)} t/s</span>
                  <span className="text-muted-foreground/30 tabular-nums shrink-0">×{m.count}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {modelEvents.length > 0 && (
          <div className="mt-2 pt-2 border-t border-border/30">
            <div className="text-muted-foreground/50 text-[9px] mb-1">Model events</div>
            <div className="space-y-0.5">
              {modelEvents.slice(0, 5).map((e, i) => (
                <div key={i} className="flex items-center justify-between gap-2 text-[10px]">
                  <span className={cn(
                    "shrink-0",
                    e.type === 'load' ? 'text-green-400' : e.type === 'unload' ? 'text-yellow-400' : e.type === 'error' ? 'text-red-400' : 'text-blue-400',
                  )}>{e.type}</span>
                  <span className="text-muted-foreground/60 truncate">{e.model}</span>
                  <span className="text-muted-foreground/30 tabular-nums shrink-0">{new Date(e.ts * 1000).toLocaleTimeString()}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {pathLats.length > 0 && (
          <div className="mt-2 pt-2 border-t border-border/30">
            <div className="text-muted-foreground/50 text-[9px] mb-1">Slowest endpoints</div>
            <div className="space-y-0.5">
              {pathLats.map((p, i) => (
                <div key={i} className="flex items-center justify-between gap-2 text-[10px]">
                  <span className="text-muted-foreground/60 truncate">{p.path}</span>
                  <span className="text-muted-foreground/40 tabular-nums shrink-0">{p.avg_ms.toFixed(0)}ms</span>
                  <span className="text-muted-foreground/30 tabular-nums shrink-0">×{p.count}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {memoryHistory.length >= 2 && (
          <div className="mt-2 pt-2 border-t border-border/30">
            <div className="flex items-center justify-between mb-1">
              <span className="text-muted-foreground/50 text-[9px]">Memory (RSS)</span>
              <Sparkline data={memoryHistory.map(m => m.rss_mb)} width={60} height={10} />
            </div>
            <div className="text-[10px] text-muted-foreground/60">
              {memoryHistory[memoryHistory.length - 1]?.rss_mb.toFixed(0)} MB
              {memoryHistory[memoryHistory.length - 1]?.system_percent ? ` (${memoryHistory[memoryHistory.length - 1]?.system_percent}% system)` : ''}
            </div>
          </div>
        )}
        {rateViolations.length > 0 && (
          <div className="mt-2 pt-2 border-t border-border/30">
            <div className="text-muted-foreground/50 text-[9px] mb-1">Rate limit hits</div>
            <div className="space-y-0.5">
              {rateViolations.slice(0, 3).map((v, i) => (
                <div key={i} className="flex items-center justify-between gap-2 text-[10px]">
                  <span className="text-yellow-400 truncate">{v.path}</span>
                  <span className="text-muted-foreground/40 tabular-nums shrink-0">{v.count}/{v.limit}/s</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {recentErrs.length > 0 && (
          <div className="mt-2 pt-2 border-t border-border/30">
            <div className="text-muted-foreground/50 text-[9px] mb-1">Server errors</div>
            <div className="space-y-0.5">
              {recentErrs.map((e, i) => (
                <div key={i} className="text-[10px] break-all leading-tight">
                  <span className="text-destructive">{e.error_type}</span>
                  <span className="text-muted-foreground/40"> {e.path}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {recentReqs.length > 0 && (
          <div className="mt-2 pt-2 border-t border-border/30">
            <div className="text-muted-foreground/50 text-[9px] mb-1">Recent requests</div>
            <div className="space-y-0.5">
              {recentReqs.map((r, i) => (
                <div key={i} className="flex items-center justify-between gap-2 text-[10px]">
                  <span className="text-muted-foreground/60 truncate">{r.method} {r.path}</span>
                  <span className={cn(
                    "shrink-0 tabular-nums",
                    r.status >= 500 ? 'text-destructive' : r.status >= 400 ? 'text-warning' : 'text-success',
                  )}>{r.status}</span>
                  <span className="text-muted-foreground/40 tabular-nums shrink-0">{r.elapsed_ms.toFixed(0)}ms</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {lastError && (
          <div className="mt-2 pt-2 border-t border-border/30">
            <div className="text-muted-foreground/50 text-[9px] mb-1">Last FE error</div>
            <div className="text-red-400 text-[10px] break-all leading-tight">{lastError.title}{lastError.requestId ? ` [${lastError.requestId}]` : ''}</div>
          </div>
        )}
      </div>
    </div>
  )
}
