'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { soulsController } from '@/lib/souls-controller'
import { useApiMonitor } from '@/lib/api-monitor-store'
import { useErrorStore } from '@/lib/error-store'
import { cn } from '@sloughgpt/strui'
import { deriveArchetype } from '@/components/souls/PersonalitySummary'
import { getUnseenCount } from '@/components/WhatsNewDialog'

function formatDuration(ms: number): string {
  const s = Math.floor(ms / 1000)
  if (s < 60) return `${s}s`
  return `${Math.floor(s / 60)}m ${s % 60}s`
}

function getFailureSummary(failures: { kind: string; timeoutMs: number; error: string; timestamp: number }[]): string {
  if (failures.length === 0) return ''
  const last = failures[0]
  const elapsed = Math.round((Date.now() - failures[failures.length - 1].timestamp) / 1000)
  if (last.kind === 'timeout') return `timeout after ${last.timeoutMs / 1000}s`
  if (last.kind === 'connection_refused') return 'connection refused'
  return last.error.slice(0, 40)
}

export function StatusBar() {
  const { health, healthLegacy, connectionStatus } = useLiveStatus()
  const recentFailures = useApiMonitor((s) => s.recentFailures)
  const failureCount = useApiMonitor((s) => s.failureCount)
  const lastOffline = useApiMonitor((s) => s.lastOffline)
  const errorCount = useErrorStore(s => s.errors.reduce((sum, e) => sum + e.count, 0))
  const [soulName, setSoulName] = useState<string | null>(null)
  const [archetypeLabel, setArchetypeLabel] = useState<string | null>(null)
  const [unseenCount, setUnseenCount] = useState(0)
  const [now, setNow] = useState(Date.now())

  useEffect(() => {
    setUnseenCount(getUnseenCount())
    const handler = () => setUnseenCount(getUnseenCount())
    window.addEventListener('whatsnew-updated', handler)
    return () => window.removeEventListener('whatsnew-updated', handler)
  }, [])

  // Tick every second for uptime display
  useEffect(() => {
    if (connectionStatus === 'connected' && health !== null) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [connectionStatus, health])

  useEffect(() => {
    if (health === null || connectionStatus !== 'connected') return
    soulsController.getCurrent().then(async (s) => {
      if (s && 'name' in s) {
        const name = (s as any).name
        setSoulName(name)
        try {
          const w = await soulsController.getTraitWeights()
          if (w && !('error' in w)) {
            const arch = deriveArchetype(w as any)
            setArchetypeLabel(arch.label)
          }
        } catch {}
      }
    }).catch(() => {})
  }, [health, connectionStatus])

  const score = health?.health_score
  const dot = connectionStatus === 'connecting' ? 'bg-muted-foreground/50' :
    connectionStatus === 'offline' ? 'bg-destructive' :
    connectionStatus === 'reloading' ? 'bg-destructive animate-pulse' :
    score != null && score < 50 ? 'bg-destructive' :
    score != null && score < 80 ? 'bg-warning' :
    health?.model_loaded ? 'bg-success' : 'bg-warning'

  // Build diagnostic status text
  let statusText: string
  if (connectionStatus === 'connecting' && health === null) {
    statusText = 'Connecting...'
  } else if (connectionStatus === 'offline') {
    const elapsed = lastOffline ? formatDuration(now - lastOffline) : ''
    const failInfo = getFailureSummary(recentFailures)
    statusText = failInfo
      ? `Offline — ${failInfo}${elapsed ? ` (${elapsed})` : ''}`
      : `Offline${elapsed ? ` (${elapsed})` : ''}`
  } else {
    statusText = health?.health_summary || (health?.model_loaded ? (health.model_type || 'Loaded') : 'No model')
  }

  return (
    <Link href="/monitoring" className="flex shrink-0 h-6 sm:h-7 items-center justify-between px-2 sm:px-3 border-t border-border/30 bg-muted/20 text-[10px] text-muted-foreground/70 hover:bg-muted/40 transition-colors" aria-label="System health and model status">
      <div className="flex items-center gap-1.5 sm:gap-3 min-w-0">
        <span className="flex items-center gap-1" aria-live="polite" aria-atomic="true">
          <span className={cn("inline-block h-1.5 w-1.5 rounded-full shrink-0", dot)} aria-hidden="true" />
          <span className="truncate max-w-[200px] sm:max-w-none" title={statusText}>{statusText}</span>
        </span>
        {connectionStatus === 'connected' && health && (
          <span className="hidden sm:inline-flex items-center gap-1 px-1 py-0.5 rounded-full bg-success/10 text-success text-[9px] font-medium leading-none">
            live
          </span>
        )}
        {failureCount > 0 && connectionStatus !== 'connected' && (
          <span className="hidden sm:inline-flex items-center px-1 py-0.5 rounded-full bg-destructive/10 text-destructive text-[9px] font-medium leading-none">
            {failureCount} fail{failureCount !== 1 ? 's' : ''}
          </span>
        )}
        {errorCount > 0 && connectionStatus === 'connected' && (
          <span className="hidden sm:inline-flex items-center px-1 py-0.5 rounded-full bg-destructive/10 text-destructive text-[9px] font-medium leading-none">
            {errorCount} err{errorCount !== 1 ? 's' : ''}
          </span>
        )}
        {soulName && archetypeLabel && (
          <span className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-primary/8 text-primary/80 text-[9px] font-medium leading-none">
            {archetypeLabel}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2 sm:gap-3">
        <button
          onClick={(e) => { e.preventDefault(); window.dispatchEvent(new CustomEvent('toggle-output-panel')) }}
          className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 rounded hover:bg-muted/60 transition-colors"
          aria-label="Toggle server output"
          title="Server output"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" /></svg>
        </button>
        <button
          onClick={(e) => { e.preventDefault(); window.dispatchEvent(new CustomEvent('toggle-whatsnew')) }}
          className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 rounded hover:bg-muted/60 transition-colors relative"
          aria-label={unseenCount > 0 ? `${unseenCount} new feature${unseenCount === 1 ? '' : 's'}` : "What's new"}
          title="What's new"
        >
          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24"><path d="M13 11h-2v-2h2v2zm0 6h-2v-6h2v6zm1-15H6v2h8V2zm4 4H2v14h16V6zM4 18V8h12v10H4z"/></svg>
          {unseenCount > 0 && (
            <span className="absolute -top-1 -right-1 min-w-[14px] h-3.5 px-1 flex items-center justify-center rounded-full bg-primary text-primary-foreground text-[9px] font-medium leading-none">
              {unseenCount > 9 ? '9+' : unseenCount}
            </span>
          )}
        </button>
        {health?.tokens_per_sec ? (
          <span className="hidden sm:inline tabular-nums">{health.tokens_per_sec.toFixed(0)} t/s</span>
        ) : health !== null && health.inference_count != null ? (
          <span className="hidden sm:inline">{health.inference_count} response{health.inference_count !== 1 ? 's' : ''}</span>
        ) : null}
      </div>
    </Link>
  )
}
