'use client'

import { useEffect, useState, useCallback } from 'react'
import Link from 'next/link'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { soulsController } from '@/lib/souls-controller'
import { useApiMonitor } from '@/lib/api-monitor-store'
import { useErrorStore } from '@/lib/error-store'
import { useErrorStream } from '@/hooks/useErrorStream'
import { cn, Button, Badge, Tooltip, TooltipTrigger, TooltipContent, IconRefresh, IconX, IconMenu, IconGrid } from '@sloughgpt/strui'
import { deriveArchetype } from '@/components/souls/PersonalitySummary'
import { getUnseenCount } from '@/components/WhatsNewDialog'
import { logger } from '@/lib/dev-log'
import { formatDuration } from '@/lib/formatDuration'
import { PUBLIC_API_URL } from '@/lib/config'

function getFailureSummary(failures: { kind: string; timeoutMs: number; error: string; timestamp: number }[]): string {
  if (failures.length === 0) return ''
  const last = failures[0]
  if (last.kind === 'timeout') return `timeout after ${last.timeoutMs / 1000}s`
  if (last.kind === 'connection_refused') return 'connection refused'
  return last.error.slice(0, 40)
}

function formatTimestamp(ts: number): string {
  const d = new Date(ts)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export function StatusBar() {
  const { health, healthLegacy, connectionStatus } = useLiveStatus()
  const recentFailures = useApiMonitor((s) => s.recentFailures)
  const failureCount = useApiMonitor((s) => s.failureCount)
  const lastOffline = useApiMonitor((s) => s.lastOffline)
  const clearFailures = useApiMonitor((s) => s.clearFailures)
  const errorCount = useErrorStore(s => s.errors.reduce((sum, e) => sum + e.count, 0))
  const { connected: errorStreamConnected, totalReceived: errorStreamTotal } = useErrorStream()
  const [soulName, setSoulName] = useState<string | null>(null)
  const [archetypeLabel, setArchetypeLabel] = useState<string | null>(null)
  const [unseenCount, setUnseenCount] = useState(0)
  const [now, setNow] = useState(Date.now())
  const [retrying, setRetrying] = useState(false)

  useEffect(() => {
    getUnseenCount().then(setUnseenCount).catch(() => setUnseenCount(0))
    const handler = () => { getUnseenCount().then(setUnseenCount).catch(() => {}) }
    window.addEventListener('whatsnew-updated', handler)
    return () => window.removeEventListener('whatsnew-updated', handler)
  }, [])

  useEffect(() => {
    if (connectionStatus === 'connected' && health !== null) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [connectionStatus, health])

  const currentSoul = health?.soul ?? null

  useEffect(() => {
    if (health === null || connectionStatus !== 'connected') return
    let cancelled = false
    soulsController.getCurrent().then(async (s) => {
      if (cancelled || !s || !('name' in s)) return
      const name = s.name
      setSoulName(name)
      try {
        const w = await soulsController.getTraitWeights()
        if (!cancelled && w && !('error' in w)) {
          const arch = deriveArchetype(w as Record<string, Record<string, number>>)
          setArchetypeLabel(arch.label)
        }
      } catch {
        logger.warning('StatusBar: failed to fetch trait weights', {})
      }
    }).catch(() => {})
    return () => { cancelled = true }
  }, [connectionStatus, currentSoul])

  const handleRetry = useCallback(async () => {
    setRetrying(true)
    try {
      const res = await fetch(`${PUBLIC_API_URL}/health`, { signal: AbortSignal.timeout(5000) })
      if (res.ok) {
        window.location.reload()
      }
    } catch {
      // Retry failed, will be picked up by existing monitor
    } finally {
      setTimeout(() => setRetrying(false), 1000)
    }
  }, [])

  const handleClearFailures = useCallback(() => {
    clearFailures()
  }, [clearFailures])

  const score = health?.health_score
  const dot = connectionStatus === 'connecting' ? 'bg-muted-foreground/50' :
    connectionStatus === 'offline' ? 'bg-destructive' :
    connectionStatus === 'reloading' ? 'bg-destructive animate-pulse' :
    score != null && score < 50 ? 'bg-destructive' :
    score != null && score < 80 ? 'bg-warning' :
    health?.model_loaded ? 'bg-success' : 'bg-warning'

  let statusText: string
  if (connectionStatus === 'connecting' && health === null) {
    statusText = 'Connecting...'
  } else if (connectionStatus === 'offline') {
    const elapsed = lastOffline ? formatDuration((now - lastOffline) / 1000) : ''
    const failInfo = getFailureSummary(recentFailures)
    statusText = failInfo
      ? `Offline — ${failInfo}${elapsed ? ` (${elapsed})` : ''}`
      : `Offline${elapsed ? ` (${elapsed})` : ''}`
  } else {
    statusText = health?.health_summary || (health?.model_loaded ? (health.model_type || 'Loaded') : 'No model')
  }

  const hasFailures = failureCount > 0 && connectionStatus !== 'connected'

  return (
    <Link href="/settings" prefetch={false} className="flex shrink-0 h-7 sm:h-8 items-center justify-between px-2 sm:px-3 border-t border-border/30 bg-muted/20 text-[10px] text-muted-foreground/70 hover:bg-muted/40 transition-colors" aria-label="System health and model status">
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
        {hasFailures && (
          <div className="flex items-center gap-0.5">
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="hidden sm:inline-flex items-center px-1 py-0.5 rounded-full bg-destructive/10 text-destructive text-[9px] font-medium leading-none cursor-help">
                  {failureCount} fail{failureCount !== 1 ? 's' : ''}
                </span>
              </TooltipTrigger>
              <TooltipContent side="top" variant="muted" className="max-w-64">
                <div className="space-y-1.5">
                  <p className="font-medium text-destructive">Connection Failures</p>
                  {recentFailures.slice(0, 3).map((f, i) => (
                    <div key={i} className="text-[10px] text-muted-foreground">
                      <span className="font-mono">{formatTimestamp(f.timestamp)}</span>
                      <span className="mx-1">·</span>
                      <span>{f.kind === 'timeout' ? `Timeout (${f.timeoutMs / 1000}s)` : f.kind === 'connection_refused' ? 'Connection refused' : f.error.slice(0, 30)}</span>
                    </div>
                  ))}
                  {recentFailures.length > 3 && (
                    <p className="text-[9px] text-muted-foreground/70">+{recentFailures.length - 3} more</p>
                  )}
                </div>
              </TooltipContent>
            </Tooltip>
            <Button
              variant="ghost"
              size="icon"
              className="h-4 w-4 p-0 text-destructive/70 hover:text-destructive hover:bg-destructive/10"
              onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleRetry() }}
              disabled={retrying}
              aria-label="Retry connection"
              title="Retry connection"
            >
              <IconRefresh className={cn("h-2.5 w-2.5", retrying && "animate-spin")} />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-4 w-4 p-0 text-muted-foreground/50 hover:text-muted-foreground hover:bg-muted/40"
              onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleClearFailures() }}
              aria-label="Clear failure count"
              title="Clear failures"
            >
              <IconX className="h-2.5 w-2.5" />
            </Button>
          </div>
        )}
        {errorCount > 0 && (
          <span className="hidden sm:inline-flex items-center px-1 py-0.5 rounded-full bg-destructive/10 text-destructive text-[9px] font-medium leading-none">
            {errorCount} err{errorCount !== 1 ? 's' : ''}
          </span>
        )}
        {errorStreamConnected && (
          <span className="hidden sm:inline-flex items-center gap-1 px-1 py-0.5 rounded-full bg-success/10 text-success text-[9px] font-medium leading-none" title={`${errorStreamTotal} error events received`}>
            <span className="inline-block h-1 w-1 rounded-full bg-green-400 animate-pulse" />
            err stream
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
          type="button"
          onClick={(e) => { e.preventDefault(); window.dispatchEvent(new CustomEvent('toggle-output-panel')) }}
          className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 rounded hover:bg-muted/60 transition-colors"
          aria-label="Toggle service output"
          title="Service output"
        >
          <IconMenu className="w-3 h-3" aria-hidden="true" />
        </button>
        <button
          type="button"
          onClick={(e) => { e.preventDefault(); window.dispatchEvent(new CustomEvent('toggle-whatsnew')) }}
          className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 rounded hover:bg-muted/60 transition-colors relative"
          aria-label={unseenCount > 0 ? `${unseenCount} new feature${unseenCount === 1 ? '' : 's'}` : "What's new"}
          title="What's new"
        >
          <IconGrid className="w-3 h-3" aria-hidden="true" />
          {unseenCount > 0 && (
            <span className="absolute -top-1 -right-1 min-w-[14px] h-3.5 px-1 flex items-center justify-center rounded-full bg-primary text-primary-foreground text-[9px] font-medium leading-none">
              {unseenCount > 9 ? '9+' : unseenCount}
            </span>
          )}
        </button>
        {health?.tokens_per_sec ? (
          <span className="hidden sm:inline-flex items-center gap-1 tabular-nums">
            <span className="inline-block h-1 w-1 rounded-full bg-success animate-pulse" aria-hidden="true" />
            {health.tokens_per_sec.toFixed(0)} tok/s
          </span>
        ) : health !== null && health.inference_count != null ? (
          <span className="hidden sm:inline tabular-nums">{health.inference_count} req{health.inference_count !== 1 ? 's' : ''}</span>
        ) : null}
        {health?.uptime_seconds != null && health.uptime_seconds > 0 && (
          <span className="hidden sm:inline tabular-nums" title="Uptime">{formatDuration(health.uptime_seconds)}</span>
        )}
      </div>
    </Link>
  )
}
