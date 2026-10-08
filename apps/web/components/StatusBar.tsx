'use client'

import { useEffect, useState, useCallback } from 'react'
import { formatTimeWithSeconds } from '@/lib/time-format'
import Link from '@/vite/next-compat/link'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { useConsciousnessStatus, getQualiaMood } from '@/hooks/useConsciousnessStatus'
import { soulsController } from '@/lib/souls-controller'
import { useApiMonitor } from '@/lib/api-monitor-store'
import { useErrorStore } from '@/lib/error-store'
import { useErrorStream } from '@/hooks/useErrorStream'
import {
  cn,
  Button,
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  IconX,
  Spinner,
} from '@sloughgpt/strui'
import { deriveArchetype } from '@/components/souls/PersonalitySummary'
import { getUnseenCount } from '@/components/WhatsNewDialog'
import { logger } from '@/lib/dev-log'
import { formatDuration } from '@/lib/formatDuration'
import { apiGet } from '@/lib/http-client'
import { useModelReadiness } from '@/lib/store'

function getFailureSummary(
  failures: { kind: string; timeoutMs: number; error: string; timestamp: number }[],
): string {
  if (failures.length === 0) return ''
  const last = failures[0]
  if (last.kind === 'timeout') return `timeout after ${last.timeoutMs / 1000}s`
  if (last.kind === 'connection_refused') return 'connection refused'
  return last.error.slice(0, 40)
}

/** `Qwen/Qwen2.5-0.5B-Instruct` -> `Qwen2.5-0.5B-Instruct` (last path segment). */
function shortModel(model: string | null): string | null {
  if (!model) return null
  const trimmed = model.trim()
  if (!trimmed) return null
  return trimmed.split('/').filter(Boolean).pop() ?? trimmed
}

/**
 * `quantization` is typed `unknown | null` because two shapes are in the wild
 * (string and `{bits}`). Read only what is understood; never render JSON at the
 * user from a payload we do not control.
 */
function quantLabel(q: unknown): string | null {
  if (typeof q === 'string') return q.trim() || null
  if (q && typeof q === 'object' && 'bits' in q) {
    const bits = (q as { bits?: unknown }).bits
    if (typeof bits === 'number' && Number.isFinite(bits)) return `${bits}-bit`
  }
  return null
}

const pctLabel = (value: number | null | undefined): string | null =>
  value == null ? null : `${Math.round(value)}%`

const msLabel = (value: number | null | undefined): string | null => {
  if (value == null || value <= 0) return null
  return value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${Math.round(value)}ms`
}

/** 494_000_000 -> `0.5B` (parameters are only ever shown formatted). */
const paramsLabel = (value: number | null | undefined): string | null => {
  if (value == null || value <= 0) return null
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`
  if (value >= 1e6) return `${Math.round(value / 1e6)}M`
  if (value >= 1e3) return `${Math.round(value / 1e3)}K`
  return String(value)
}

export function StatusBar() {
  const { health, healthLegacy, connectionStatus } = useLiveStatus()
  const recentFailures = useApiMonitor((s) => s.recentFailures)
  const failureCount = useApiMonitor((s) => s.failureCount)
  const lastOffline = useApiMonitor((s) => s.lastOffline)
  const clearFailures = useApiMonitor((s) => s.clearFailures)
  const errorCount = useErrorStore((s) => s.errors.reduce((sum, e) => sum + e.count, 0))
  const { connected: errorStreamConnected, totalReceived: errorStreamTotal } = useErrorStream()
  const modelReadiness = useModelReadiness()
  const [soulName, setSoulName] = useState<string | null>(null)
  const [archetypeLabel, setArchetypeLabel] = useState<string | null>(null)
  const [unseenCount, setUnseenCount] = useState(0)
  const [now, setNow] = useState(Date.now())
  const [retrying, setRetrying] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const { status: consciousnessStatus } = useConsciousnessStatus()

  useEffect(() => {
    getUnseenCount()
      .then(setUnseenCount)
      .catch(() => setUnseenCount(0))
    const handler = () => {
      getUnseenCount()
        .then(setUnseenCount)
        .catch(() => {})
    }
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
    soulsController
      .getCurrent()
      .then(async (s) => {
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
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [connectionStatus, currentSoul])

  const handleRetry = useCallback(async () => {
    setRetrying(true)
    try {
      await apiGet('/health', undefined, { signal: AbortSignal.timeout(5000), silent: true })
      window.location.reload()
    } catch {
      /* will be picked up by existing monitor */
    } finally {
      setTimeout(() => setRetrying(false), 1000)
    }
  }, [])

  const handleClearFailures = useCallback(() => {
    clearFailures()
  }, [clearFailures])

  const score = health?.health_score
  const dotColor =
    connectionStatus === 'connecting'
      ? 'bg-muted-foreground/50'
      : connectionStatus === 'offline'
        ? 'bg-destructive'
        : connectionStatus === 'reloading'
          ? 'bg-destructive animate-pulse'
          : score != null && score < 50
            ? 'bg-destructive'
            : score != null && score < 80
              ? 'bg-warning'
              : health?.model_loaded
                ? 'bg-success'
                : 'bg-warning'

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
    statusText =
      health?.health_summary || (health?.model_loaded ? health.model_type || 'Loaded' : 'No model')
  }

  const hasFailures = failureCount > 0 && connectionStatus !== 'connected'
  const isOffline = connectionStatus === 'offline'
  const isOnline = connectionStatus === 'connected' && health !== null

  // Rich strip — derived once, rendered as chips. Everything here is a
  // *summary*: the raw numbers live in the expanded row and the monitoring page.
  const engine = shortModel(health?.model_type ?? null)
  const quant = quantLabel(health?.quantization ?? null)
  const device = health?.device ?? null
  const params = paramsLabel(health?.num_parameters)
  const cpu = pctLabel(health?.cpu_percent)
  const ram = pctLabel(health?.memory_percent)
  const p50 = msLabel(health?.avg_latency_ms)
  const p95 = msLabel(health?.p95_latency_ms)
  const rpm = health?.requests_per_minute ?? 0
  const pool = health?.training_pool ?? null
  const activeJobs = pool?.active_jobs ?? 0
  // The health summary is a sentence that frequently already names the model —
  // never say it twice on one line.
  const showEngine = Boolean(engine) && !statusText.includes(engine ?? '')

  return (
    <div className="sl-status-bar">
      {/* Compact row — always visible */}
      <div className="sl-status-bar-row">
        {/* Left: Status dot + text. The text span — and only that span — is the
            live region: announcing the whole bar would re-read every metric on
            each health tick. */}
        <Link
          href="/settings"
          prefetch={false}
          className="sl-status-bar-status"
          aria-label="System health"
        >
          <span className={cn('sl-status-bar-dot', dotColor)} aria-hidden="true" />
          <span
            role="status"
            aria-live="polite"
            aria-atomic="true"
            className="truncate max-w-[180px] sm:max-w-none"
            title={statusText}
          >
            {statusText}
          </span>
          {isOnline && <span className="sl-badge sl-badge-success">live</span>}
        </Link>

        {/* Right: identity · failures · live metrics · controls */}
        <div className="sl-status-bar-actions">
          {/* Engine identity — stable while the metrics move, so it leads. */}
          {showEngine && (
            <span
              className="sl-badge sl-badge-secondary sl-status-bar-engine"
              title={[engine, quant, device].filter(Boolean).join(' · ')}
            >
              {engine}
              {quant && <span className="opacity-70">{quant}</span>}
            </span>
          )}

          {/* Failure count */}
          {hasFailures && (
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="sl-badge sl-badge-destructive cursor-help" role="status">
                  {failureCount} fail{failureCount !== 1 ? 's' : ''}
                </span>
              </TooltipTrigger>
              <TooltipContent side="top" variant="muted" className="max-w-64">
                <div className="space-y-1.5">
                  <p className="font-medium text-destructive">Connection Failures</p>
                  {recentFailures.slice(0, 3).map((f, i) => (
                    <div key={i} className="text-[10px] text-muted-foreground">
                      <span className="font-mono">{formatTimeWithSeconds(f.timestamp)}</span>
                      <span className="mx-1">·</span>
                      <span>
                        {f.kind === 'timeout'
                          ? `Timeout (${f.timeoutMs / 1000}s)`
                          : f.kind === 'connection_refused'
                            ? 'Connection refused'
                            : f.error.slice(0, 30)}
                      </span>
                    </div>
                  ))}
                  {recentFailures.length > 3 && (
                    <p className="text-[9px] text-muted-foreground/70">
                      +{recentFailures.length - 3} more
                    </p>
                  )}
                </div>
              </TooltipContent>
            </Tooltip>
          )}

          {/* Error count */}
          {errorCount > 0 && (
            <span className="sl-badge sl-badge-destructive">
              {errorCount} err{errorCount !== 1 ? 's' : ''}
            </span>
          )}

          {/* Error stream */}
          {errorStreamConnected && (
            <span
              className="sl-badge sl-badge-success"
              title={`${errorStreamTotal} error events received`}
            >
              <span className="h-1 w-1 rounded-full bg-success animate-pulse" />
              stream
            </span>
          )}

          {/* Soul archetype */}
          {soulName && archetypeLabel && (
            <span className="sl-badge sl-badge-primary">{archetypeLabel}</span>
          )}

          {/* Consciousness status */}
          {consciousnessStatus && consciousnessStatus.enabled && (
            <span
              className="sl-badge sl-badge-secondary"
              title={`Consciousness: ${consciousnessStatus.level}/3 | Mood: ${getQualiaMood(consciousnessStatus.current_qualia)}`}
            >
              <span className="h-1 w-1 rounded-full bg-primary animate-pulse" aria-hidden="true" />
              {getQualiaMood(consciousnessStatus.current_qualia) || `L${consciousnessStatus.level}`}
            </span>
          )}

          {/* Live metrics — one cluster, hairline-separated from identity.
              Values only; the sentence lives on the left, the full breakdown
              in the expanded row. */}
          {isOnline && (
            <div className="sl-status-bar-cluster" role="group" aria-label="Live metrics">
              {health?.is_inferencing && (
                <span className="sl-badge sl-badge-primary" title="Generating">
                  <span
                    className="h-1 w-1 rounded-full bg-primary animate-pulse"
                    aria-hidden="true"
                  />
                  gen
                </span>
              )}

              {health?.tokens_per_sec ? (
                <span
                  className="sl-status-bar-metric tabular-nums"
                  title={`${health.tokens_per_sec.toFixed(1)} tokens/second`}
                >
                  <span
                    className="h-1 w-1 rounded-full bg-success animate-pulse"
                    aria-hidden="true"
                  />
                  {health.tokens_per_sec.toFixed(0)} tok/s
                </span>
              ) : health?.inference_count != null ? (
                <span
                  className="sl-status-bar-metric tabular-nums"
                  title={`${health.inference_count} requests served`}
                >
                  {health.inference_count} req{health.inference_count !== 1 ? 's' : ''}
                </span>
              ) : null}

              {p95 && (
                <span
                  className="sl-status-bar-metric tabular-nums"
                  title={`Latency p95${p50 ? ` · avg ${p50}` : ''}`}
                >
                  {p95} p95
                </span>
              )}
              {cpu && (
                <span className="sl-status-bar-metric tabular-nums" title="Host CPU">
                  {cpu} CPU
                </span>
              )}
              {ram && (
                <span className="sl-status-bar-metric tabular-nums" title="Host memory">
                  {ram} RAM
                </span>
              )}
              {activeJobs > 0 && (
                <span
                  className="sl-status-bar-metric tabular-nums"
                  title={`Training ${activeJobs} of ${pool?.max_workers ?? '?'} workers`}
                >
                  <span
                    className="h-1 w-1 rounded-full bg-primary animate-pulse"
                    aria-hidden="true"
                  />
                  {activeJobs} train
                </span>
              )}
            </div>
          )}

          {/* What's new — always visible */}
          <button
            type="button"
            onClick={(e) => {
              e.preventDefault()
              window.dispatchEvent(new CustomEvent('toggle-whatsnew'))
            }}
            className="sl-status-bar-btn relative"
            aria-label={
              unseenCount > 0
                ? `${unseenCount} new feature${unseenCount === 1 ? '' : 's'}`
                : "What's new"
            }
            title="What's new"
          >
            <svg
              className="h-3 w-3"
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="8" cy="8" r="6" />
              <path d="M8 5v3l2 1" />
            </svg>
            {unseenCount > 0 && (
              <span className="absolute -top-1.5 -right-1.5 min-w-[14px] h-3.5 px-1 flex items-center justify-center rounded-full bg-primary text-primary-foreground text-[8px] font-medium leading-none">
                {unseenCount > 9 ? '9+' : unseenCount}
              </span>
            )}
          </button>

          {/* Expand toggle */}
          <button
            type="button"
            onClick={(e) => {
              e.preventDefault()
              e.stopPropagation()
              setExpanded(!expanded)
            }}
            className="sl-status-bar-btn"
            aria-label={expanded ? 'Collapse details' : 'Expand details'}
            aria-expanded={expanded}
          >
            <svg
              className={cn('h-3 w-3 transition-transform', expanded && 'rotate-180')}
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M4 6l4 4 4-4" />
            </svg>
          </button>
        </div>
      </div>

      {/* Expanded row — details */}
      {expanded && (
        <div className="sl-status-bar-details">
          {/* Engine */}
          <div className="sl-status-bar-detail-group">
            <span className="sl-status-bar-detail-label">Model</span>
            <span className="sl-status-bar-detail-value">
              {engine || health?.model_type || '—'}
            </span>
          </div>
          {quant && (
            <div className="sl-status-bar-detail-group">
              <span className="sl-status-bar-detail-label">Quant</span>
              <span className="sl-status-bar-detail-value">{quant}</span>
            </div>
          )}
          {device && (
            <div className="sl-status-bar-detail-group">
              <span className="sl-status-bar-detail-label">Device</span>
              <span className="sl-status-bar-detail-value">{device}</span>
            </div>
          )}
          {params && (
            <div className="sl-status-bar-detail-group">
              <span className="sl-status-bar-detail-label">Params</span>
              <span className="sl-status-bar-detail-value tabular-nums">{params}</span>
            </div>
          )}

          {/* Performance */}
          {health?.tokens_per_sec ? (
            <div className="sl-status-bar-detail-group">
              <span className="sl-status-bar-detail-label">Speed</span>
              <span className="sl-status-bar-detail-value tabular-nums">
                {health.tokens_per_sec.toFixed(0)} tok/s
              </span>
            </div>
          ) : null}
          {p50 && (
            <div className="sl-status-bar-detail-group">
              <span className="sl-status-bar-detail-label">Avg</span>
              <span className="sl-status-bar-detail-value tabular-nums">{p50}</span>
            </div>
          )}
          {p95 && (
            <div className="sl-status-bar-detail-group">
              <span className="sl-status-bar-detail-label">P95</span>
              <span className="sl-status-bar-detail-value tabular-nums">{p95}</span>
            </div>
          )}
          {rpm > 0 && (
            <div className="sl-status-bar-detail-group">
              <span className="sl-status-bar-detail-label">Rate</span>
              <span className="sl-status-bar-detail-value tabular-nums">{rpm}/min</span>
            </div>
          )}
          <div className="sl-status-bar-detail-group">
            <span className="sl-status-bar-detail-label">Requests</span>
            <span className="sl-status-bar-detail-value tabular-nums">
              {health?.request_count ?? '—'}
            </span>
          </div>
          <div className="sl-status-bar-detail-group">
            <span className="sl-status-bar-detail-label">Errors</span>
            <span
              className={cn(
                'sl-status-bar-detail-value tabular-nums',
                (health?.error_count ?? 0) > 0 && 'text-destructive',
              )}
            >
              {health?.error_count ?? '—'}
            </span>
          </div>

          {/* Host */}
          {cpu && (
            <div className="sl-status-bar-detail-group">
              <span className="sl-status-bar-detail-label">CPU</span>
              <span className="sl-status-bar-detail-value tabular-nums">{cpu}</span>
            </div>
          )}
          {ram && (
            <div className="sl-status-bar-detail-group">
              <span className="sl-status-bar-detail-label">Memory</span>
              <span className="sl-status-bar-detail-value tabular-nums">{ram}</span>
            </div>
          )}
          <div className="sl-status-bar-detail-group">
            <span className="sl-status-bar-detail-label">Uptime</span>
            <span className="sl-status-bar-detail-value tabular-nums">
              {health?.uptime_seconds ? formatDuration(health.uptime_seconds) : '—'}
            </span>
          </div>
          {health?.health_score != null && (
            <div className="sl-status-bar-detail-group">
              <span className="sl-status-bar-detail-label">Score</span>
              <span className="sl-status-bar-detail-value tabular-nums">{health.health_score}</span>
            </div>
          )}
          {activeJobs > 0 && pool && (
            <div className="sl-status-bar-detail-group">
              <span className="sl-status-bar-detail-label">Training</span>
              <span className="sl-status-bar-detail-value tabular-nums">
                {activeJobs}/{pool.max_workers}
              </span>
            </div>
          )}
          {modelReadiness && !modelReadiness.ready && (
            <div className="sl-status-bar-detail-group">
              <span className="sl-status-bar-detail-label">Loading</span>
              <span className="sl-status-bar-detail-value text-warning">
                {modelReadiness.message}
              </span>
            </div>
          )}
          <div className="sl-status-bar-detail-spacer" />
          {/* Retry & Clear buttons (only when offline/errored) */}
          {isOffline && (
            <Button
              variant="ghost"
              size="icon"
              className="h-5 w-5 p-0 text-destructive/70 hover:text-destructive"
              onClick={handleRetry}
              disabled={retrying}
              aria-label="Retry connection"
              title="Retry"
            >
              <Spinner className="h-3 w-3" />
            </Button>
          )}
          {hasFailures && (
            <Button
              variant="ghost"
              size="icon"
              className="h-5 w-5 p-0 text-muted-foreground/50 hover:text-muted-foreground"
              onClick={handleClearFailures}
              aria-label="Clear failures"
              title="Clear"
            >
              <IconX className="h-3 w-3" />
            </Button>
          )}
          {/* Output panel */}
          <button
            type="button"
            onClick={(e) => {
              e.preventDefault()
              window.dispatchEvent(new CustomEvent('toggle-output-panel'))
            }}
            className="sl-status-bar-btn"
            aria-label="Toggle service output"
            title="Service output"
          >
            <svg
              className="h-3 w-3"
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M2 4h12M2 8h12M2 12h8" />
            </svg>
          </button>
        </div>
      )}
    </div>
  )
}
