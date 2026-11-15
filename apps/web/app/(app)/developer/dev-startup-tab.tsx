'use client'

/** Startup tab — live stage timeline and history chart. */
import { StartupTimeline } from '@/components/startup/StartupTimeline'
import { StartupHistoryChart, useStartupHistory } from '@/components/startup/StartupHistoryChart'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { useQuery } from '@/lib/cache'
import { authFetch } from '@/lib/http-client'
import { StatusDot, cn } from '@sloughgpt/strui'
import { useEffect } from 'react'

async function fetchStartupProgress(): Promise<Record<string, unknown>> {
  const res = await authFetch('/health/startup-progress')
  const json = await res.json()
  return (json.data ?? {}) as Record<string, unknown>
}

export function StartupTab() {
  const {
    health,
    startupStage,
    startupElapsed,
    startupModelProgress,
    startupModelProgressMessage,
  } = useLiveStatus()
  // Both GETs are cache-backed now, so `r` (revalidateAll) re-reads this tab in
  // place. The history read shares its key with <StartupHistoryChart/>, which
  // renders the same endpoint — one fetch, one revalidation, two views. Keys are
  // plain strings so they are referentially stable: `refetch` is
  // `useCallback([key])`, and stable `refetch` identities keep the poll below
  // from being re-created on every render.
  const { data: startupData, refetch: refetchStartupProgress } = useQuery<Record<string, unknown>>(
    'developer-startup-progress',
    fetchStartupProgress,
  )
  const { data: historyData, refetch: refetchStartupHistory } = useStartupHistory()

  // The 2s poll stays a plain effect: it is a timer, i.e. a side effect, not a
  // data read. The reads above are cached, so a refresh reaches this tab without
  // the component ever remounting.
  useEffect(() => {
    const poll = () => {
      void refetchStartupProgress()
      void refetchStartupHistory()
    }
    const interval = setInterval(poll, 2000)
    return () => clearInterval(interval)
  }, [refetchStartupProgress, refetchStartupHistory])

  const STAGE_COLORS: Record<string, string> = {
    init: 'text-muted-foreground/60',
    critical: 'text-warning',
    ready: 'text-info',
    background: 'text-success',
  }

  const HOOK_STATUS_COLORS: Record<string, string> = {
    pending: 'text-muted-foreground/60',
    running: 'text-warning',
    ok: 'text-success',
    timeout: 'text-destructive',
    error: 'text-destructive',
  }

  const hooks = startupData?.hooks as
    | Record<
        string,
        {
          name: string
          stage: string
          status: string
          duration_seconds: number
          error: string | null
        }
      >
    | undefined
  const stages = startupData?.stages as
    Record<string, { hooks: string[]; time: number | null }> | undefined
  const stats = historyData?.stats as
    | {
        count: number
        avg_duration: number
        min_duration: number
        max_duration: number
        p50_duration: number
        p95_duration: number
        success_rate: number
      }
    | undefined
  const stageStats = historyData?.stage_stats as
    | Record<string, { avg: number; min: number; max: number; p50: number; count: number }>
    | undefined
  const alerts = historyData?.alerts as
    Array<{ type: string; severity: string; message: string; timestamp: number }> | undefined
  const suggestions = historyData?.suggestions as
    | Array<{ type: string; severity: string; message: string; stage?: string; hook?: string }>
    | undefined

  return (
    <div className="space-y-4">
      {/* Alerts */}
      {alerts && alerts.length > 0 && (
        <div className="space-y-2">
          {alerts.map((alert, i) => (
            <div
              key={`${alert.type}-${i}`}
              className={cn(
                'rounded-xl border px-4 py-3 flex items-center gap-3',
                alert.severity === 'error'
                  ? 'border-destructive/20 bg-destructive/[0.06]'
                  : 'border-warning/20 bg-warning/[0.06]',
              )}
            >
              <StatusDot tone={alert.severity === 'error' ? 'destructive' : 'warning'} />
              <div className="flex-1 min-w-0">
                <span
                  className={cn(
                    'text-[11px] font-medium',
                    alert.severity === 'error' ? 'text-destructive' : 'text-warning',
                  )}
                >
                  {alert.type.replace(/_/g, ' ')}
                </span>
                <span className="text-[10px] text-muted-foreground/60 ml-2">{alert.message}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Suggestions */}
      {suggestions && suggestions.length > 0 && (
        <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
          <div className="h-9 px-4 bg-muted/30 border-b border-border/20 flex items-center">
            <span className="text-[11px] font-medium text-muted-foreground/80">
              Optimization Suggestions
            </span>
            <span className="text-[9px] text-muted-foreground/60 ml-2">
              {suggestions.length} suggestions
            </span>
          </div>
          <div className="divide-y divide-white/[0.04]">
            {suggestions.map((suggestion, i) => (
              <div key={`${suggestion.type}-${i}`} className="px-4 py-3 flex items-start gap-3">
                <StatusDot
                  tone={
                    suggestion.severity === 'error'
                      ? 'destructive'
                      : suggestion.severity === 'warning'
                        ? 'warning'
                        : 'primary'
                  }
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        'text-[11px] font-medium',
                        suggestion.severity === 'error'
                          ? 'text-destructive'
                          : suggestion.severity === 'warning'
                            ? 'text-warning'
                            : 'text-info',
                      )}
                    >
                      {suggestion.type.replace(/_/g, ' ')}
                    </span>
                    {suggestion.stage && (
                      <span className="text-[9px] text-muted-foreground/60 bg-muted/30 px-1.5 py-0.5 rounded">
                        {suggestion.stage}
                      </span>
                    )}
                    {suggestion.hook && (
                      <span className="text-[9px] text-muted-foreground/60 bg-muted/30 px-1.5 py-0.5 rounded">
                        {suggestion.hook}
                      </span>
                    )}
                  </div>
                  <span className="text-[10px] text-muted-foreground/80">{suggestion.message}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* History Chart */}
      <StartupHistoryChart />

      {/* Overview */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="rounded-xl border border-border/20 bg-card p-4">
          <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
            Stage
          </div>
          <div
            className={cn(
              'text-[14px] font-medium font-mono',
              STAGE_COLORS[startupStage] ?? 'text-muted-foreground',
            )}
          >
            {startupStage}
          </div>
        </div>
        <div className="rounded-xl border border-border/20 bg-card p-4">
          <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
            Elapsed
          </div>
          <div className="text-[14px] font-medium font-mono text-muted-foreground">
            {startupElapsed.toFixed(1)}s
          </div>
        </div>
        <div className="rounded-xl border border-border/20 bg-card p-4">
          <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
            Model
          </div>
          <div className="text-[14px] font-medium font-mono text-muted-foreground">
            {Math.round(startupModelProgress * 100)}%
          </div>
        </div>
        <div className="rounded-xl border border-border/20 bg-card p-4">
          <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
            Status
          </div>
          <div
            className={cn(
              'text-[14px] font-medium',
              startupStage === 'background' ? 'text-success' : 'text-warning',
            )}
          >
            {startupStage === 'background' ? 'Ready' : 'Starting'}
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="rounded-xl border border-border/20 bg-card p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] text-muted-foreground/80">Startup Progress</span>
          <span className="text-[11px] text-muted-foreground/60 font-mono">
            {startupModelProgressMessage}
          </span>
        </div>
        <div className="h-2 rounded-full bg-muted/30 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-info to-success transition-all duration-300"
            style={{
              width: `${Math.min(100, (startupModelProgress || (startupStage === 'background' ? 1 : startupStage === 'ready' ? 0.6 : startupStage === 'critical' ? 0.3 : 0.1)) * 100)}%`,
            }}
          />
        </div>
        <div className="flex justify-between mt-1">
          {['init', 'critical', 'ready', 'background'].map((stage, i) => (
            <span
              key={stage}
              className={cn(
                'text-[9px] font-mono',
                i <= ['init', 'critical', 'ready', 'background'].indexOf(startupStage)
                  ? 'text-success'
                  : 'text-foreground',
              )}
            >
              {stage}
            </span>
          ))}
        </div>
      </div>

      {/* Performance comparison */}
      {stats && stats.count > 0 && (
        <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
          <div className="h-9 px-4 bg-muted/30 border-b border-border/20 flex items-center">
            <span className="text-[11px] font-medium text-muted-foreground/80">
              Performance History
            </span>
            <span className="text-[9px] text-muted-foreground/60 ml-2">
              {stats.count} startups recorded
            </span>
          </div>
          <div className="p-4">
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
              <div className="text-center">
                <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
                  Avg
                </div>
                <div className="text-[14px] font-mono text-muted-foreground">
                  {stats.avg_duration}s
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
                  P50
                </div>
                <div className="text-[14px] font-mono text-muted-foreground">
                  {stats.p50_duration}s
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
                  P95
                </div>
                <div className="text-[14px] font-mono text-muted-foreground">
                  {stats.p95_duration}s
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
                  Min
                </div>
                <div className="text-[14px] font-mono text-success">{stats.min_duration}s</div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
                  Max
                </div>
                <div className="text-[14px] font-mono text-destructive">{stats.max_duration}s</div>
              </div>
            </div>
            {startupElapsed > 0 && stats.avg_duration > 0 && (
              <div className="mt-3 pt-3 border-t border-white/[0.04] flex items-center justify-center gap-2">
                <span className="text-[10px] text-muted-foreground/60">Current:</span>
                <span
                  className={cn(
                    'text-[11px] font-mono font-medium',
                    startupElapsed < stats.avg_duration ? 'text-success' : 'text-warning',
                  )}
                >
                  {startupElapsed.toFixed(1)}s
                </span>
                <span className="text-[10px] text-muted-foreground/60">
                  vs avg {stats.avg_duration}s
                </span>
                {startupElapsed < stats.avg_duration && (
                  <span className="text-[9px] text-success">
                    ({((1 - startupElapsed / stats.avg_duration) * 100).toFixed(0)}% faster)
                  </span>
                )}
                {startupElapsed > stats.avg_duration && (
                  <span className="text-[9px] text-warning">
                    (+{((startupElapsed / stats.avg_duration - 1) * 100).toFixed(0)}% slower)
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Stages */}
      {stages && (
        <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
          <div className="h-9 px-4 bg-muted/30 border-b border-border/20 flex items-center">
            <span className="text-[11px] font-medium text-muted-foreground/80">Stages</span>
          </div>
          <div className="divide-y divide-white/[0.04]">
            {Object.entries(stages).map(([stage, data]) => (
              <div key={stage} className="px-4 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <StatusDot
                    tone={
                      stage === startupStage ? 'primary' : data.time !== null ? 'success' : 'muted'
                    }
                    pulse={stage === startupStage}
                  />
                  <span className="text-[12px] font-medium text-muted-foreground capitalize">
                    {stage}
                  </span>
                  <span className="text-[10px] text-muted-foreground/60 font-mono">
                    {data.hooks.length} hooks
                  </span>
                </div>
                <span className="text-[11px] font-mono text-muted-foreground/60">
                  {data.time !== null ? `${data.time}s` : '—'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Stage Stats */}
      {stageStats && Object.keys(stageStats).length > 0 && (
        <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
          <div className="h-9 px-4 bg-muted/30 border-b border-border/20 flex items-center">
            <span className="text-[11px] font-medium text-muted-foreground/80">
              Stage Performance
            </span>
          </div>
          <div className="p-4">
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
              {Object.entries(stageStats).map(([stage, data]) => (
                <div key={stage} className="rounded-lg border border-white/[0.04] bg-muted/20 p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <StatusDot tone={stage === startupStage ? 'primary' : 'success'} />
                    <span className="text-[11px] font-medium text-muted-foreground capitalize">
                      {stage}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[10px]">
                    <div>
                      <span className="text-muted-foreground/60">Avg</span>
                      <span className="ml-1 font-mono text-muted-foreground">{data.avg}s</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground/60">P50</span>
                      <span className="ml-1 font-mono text-muted-foreground">{data.p50}s</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground/60">Min</span>
                      <span className="ml-1 font-mono text-success">{data.min}s</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground/60">Max</span>
                      <span className="ml-1 font-mono text-destructive">{data.max}s</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Timeline */}
      <StartupTimeline />

      {/* Hooks */}
      {hooks && Object.keys(hooks).length > 0 && (
        <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
          <div className="h-9 px-4 bg-muted/30 border-b border-border/20 flex items-center">
            <span className="text-[11px] font-medium text-muted-foreground/80">Hooks</span>
          </div>
          <div className="divide-y divide-white/[0.04]">
            {Object.values(hooks).map((hook) => (
              <div key={hook.name} className="px-4 py-2.5 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <StatusDot
                    tone={
                      hook.status === 'ok'
                        ? 'success'
                        : hook.status === 'running'
                          ? 'warning'
                          : hook.status === 'error' || hook.status === 'timeout'
                            ? 'destructive'
                            : 'muted'
                    }
                    pulse={hook.status === 'running'}
                  />
                  <span className="text-[11px] font-mono text-muted-foreground">{hook.name}</span>
                  <span className="text-[9px] text-muted-foreground/60 uppercase">
                    {hook.stage}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  {hook.error && (
                    <span className="text-[9px] text-destructive max-w-32 truncate">
                      {hook.error}
                    </span>
                  )}
                  <span className="text-[10px] font-mono text-muted-foreground/60">
                    {hook.duration_seconds > 0 ? `${hook.duration_seconds}s` : '—'}
                  </span>
                  <span
                    className={cn(
                      'text-[9px] font-mono uppercase',
                      HOOK_STATUS_COLORS[hook.status],
                    )}
                  >
                    {hook.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
