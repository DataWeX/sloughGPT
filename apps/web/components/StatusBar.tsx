'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useApiHealth } from '@/hooks/useApiHealth'
import { soulsController } from '@/lib/souls-controller'
import { PUBLIC_API_URL } from '@/lib/config'
import { cn } from '@/lib/cn'

const API = PUBLIC_API_URL || 'http://localhost:8000'

interface HealthSummary {
  score: number
  status: string
  summary: string
  model_loaded: boolean
  model_type: string | null
  soul: string | null
  uptime_seconds: number
  request_count: number
  error_count: number
  tokens_per_sec: number
  cpu_percent: number | null
  memory_percent: number | null
}

export function StatusBar() {
  const { state: health } = useApiHealth()
  const [soulName, setSoulName] = useState<string | null>(null)
  const [summary, setSummary] = useState<HealthSummary | null>(null)

  useEffect(() => {
    if (health === null || health === 'offline') return
    soulsController.getCurrent().then(s => {
      if (s && 'name' in s) setSoulName((s as any).name)
    }).catch(() => {})
  }, [health])

  // Poll /health/summary every 10s for score
  useEffect(() => {
    let active = true
    const poll = async () => {
      try {
        const r = await fetch(`${API}/health/summary`, { signal: AbortSignal.timeout(5000) })
        if (r.ok && active) setSummary(await r.json())
      } catch {}
    }
    poll()
    const id = setInterval(poll, 10000)
    return () => { active = false; clearInterval(id) }
  }, [])

  const score = summary?.score
  const dot = health === null ? 'bg-muted-foreground/50' :
    health === 'offline' ? 'bg-destructive' :
    score != null && score < 50 ? 'bg-destructive' :
    score != null && score < 80 ? 'bg-warning' :
    health.model_loaded ? 'bg-success' : 'bg-warning'

  // Use the human-readable summary from the flow pipeline
  const statusText = health === null ? 'Connecting...' :
    health === 'offline' ? 'Offline' :
    summary?.summary || (health.model_loaded ? health.model_type : 'No model')

  return (
    <Link href="/monitoring" className="flex shrink-0 h-6 sm:h-7 items-center justify-between px-2 sm:px-3 border-t border-border/30 bg-muted/20 text-[10px] text-muted-foreground/70 hover:bg-muted/40 transition-colors" aria-label="System health and model status">
      <div className="flex items-center gap-1.5 sm:gap-3 min-w-0">
        <span className="flex items-center gap-1" aria-live="polite" aria-atomic="true">
          <span className={cn("inline-block h-1.5 w-1.5 rounded-full shrink-0", dot)} aria-hidden="true" />
          <span className="truncate max-w-[200px] sm:max-w-none">{statusText}</span>
        </span>
      </div>
      <div className="flex items-center gap-2 sm:gap-3">
        {summary?.tokens_per_sec ? (
          <span className="hidden sm:inline tabular-nums">{summary.tokens_per_sec.toFixed(0)} t/s</span>
        ) : health !== null && health !== 'offline' && health.inference_count != null ? (
          <span className="hidden sm:inline">{health.inference_count} response{health.inference_count !== 1 ? 's' : ''}</span>
        ) : null}
      </div>
    </Link>
  )
}
