'use client'

import { useState, useEffect, useRef } from 'react'
import { useErrorStore } from '@/lib/error-store'
import { useLiveStatus } from '@/hooks/useLiveStatus'

export interface HealthScore {
  score: number
  status: string
  error_rate_score: number
  latency_score: number
  throughput_score: number
  uptime_score: number
}

export interface ModelMetric {
  model: string
  count: number
  total_tokens: number
  tokens_per_sec: number
  avg_tokens: number
}

export interface BackendDebug {
  recent_requests: Array<{ path: string; method: string; status: number; elapsed_ms: number }>
  gpu_backend: string | null
}

export function scoreColor(score: number): string {
  if (score >= 80) return 'text-green-400'
  if (score >= 50) return 'text-yellow-400'
  return 'text-red-400'
}

export function scoreBg(score: number): string {
  if (score >= 80) return 'bg-green-400/10 border-green-400/20'
  if (score >= 50) return 'bg-yellow-400/10 border-yellow-400/20'
  return 'bg-red-400/10 border-red-400/20'
}

export interface UseDebugOverlayReturn {
  errors: Array<{ message: string; timestamp: number }>
  health: HealthScore | null
  gpuBackend: string | null
  recentReqs: BackendDebug['recent_requests']
  frontendErrCount: number
  backendHealth: { status: string; latency_ms?: number } | null
  backendErrors: number
  avgLatency: number
  requestsPerMin: number
  modelMetrics: ModelMetric[]
}

export function useDebugOverlay(): UseDebugOverlayReturn {
  const errors = useErrorStore(s => s.errors)
  const { health } = useLiveStatus()
  const [gpuBackend, setGpuBackend] = useState<string | null>(null)
  const [recentReqs, setRecentReqs] = useState<BackendDebug['recent_requests']>([])
  const timerRef = useRef<ReturnType<typeof setInterval>>()

  useEffect(() => {
    let active = true
    const refresh = async () => {
      try {
        const r = await fetch('/health/debug', { signal: AbortSignal.timeout(3000) })
        if (active && r.ok) {
          const d = await r.json()
          setGpuBackend(d.gpu_backend ?? null)
          setRecentReqs(d.recent_requests ?? [])
        }
      } catch {
        // ignore
      }
    }
    refresh()
    timerRef.current = setInterval(refresh, 5000)
    return () => { if (timerRef.current) clearInterval(timerRef.current); active = false }
  }, [])

  const h = health
  const frontendErrCount = errors.length
  const backendHealth = h ? { status: h.status, latency_ms: undefined } : null
  const backendErrors = h ? Math.round((1 - h.error_rate_score / 100) * 100) : 0
  const avgLatency = recentReqs.length > 0
    ? recentReqs.reduce((s, r) => s + r.elapsed_ms, 0) / recentReqs.length
    : 0
  const requestsPerMin = recentReqs.length

  const modelMetrics: ModelMetric[] = []

  return {
    errors, health: h, gpuBackend, recentReqs,
    frontendErrCount, backendHealth, backendErrors,
    avgLatency, requestsPerMin, modelMetrics,
  }
}
