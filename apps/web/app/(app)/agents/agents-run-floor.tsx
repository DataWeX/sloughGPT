'use client'

/**
 * Factory floor — live run stations (ux/agents "Watch the Factory").
 *
 * Reads the single-writer run-stream store (SSE frames → consume,
 * GET /agents/runs → hydrate); it never talks to the network itself.
 * Loading is the explicit `runsLoading` boolean — never inferred from
 * empty stations (an empty floor means "no runs", not "still loading").
 *
 * Vocabulary: docs/routers.md "Run-stream event vocabulary". Stations are
 * queued | live | done | failed; every status shows as text, never color alone.
 */
import { EmptyCard, Skeleton, cn } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { useRunStreamStore } from './use-run-stream-store'
import type { StationStatus } from './run-state'

function stationDot(status: StationStatus): string {
  if (status === 'done') return 'bg-success'
  if (status === 'failed') return 'bg-destructive'
  if (status === 'live') return 'bg-warning animate-pulse'
  return 'bg-muted-foreground/30'
}

function stationBadge(status: StationStatus): string {
  if (status === 'done') return 'bg-success/15 text-success'
  if (status === 'failed') return 'bg-destructive/15 text-destructive'
  if (status === 'live') return 'bg-warning/15 text-warning'
  return 'bg-muted text-muted-foreground'
}

function phaseLabel(phase: string, level: number, totalLevels: number): string {
  switch (phase) {
    case 'PLAN':
      return 'Planning subtasks…'
    case 'EXECUTE':
      return totalLevels > 0
        ? `Executing level ${level}/${totalLevels}…`
        : 'Executing…'
    case 'COMPOSE':
      return 'Composing final response…'
    case 'COMPLETE':
      return 'Run complete'
    case 'ERROR':
      return 'Run failed'
    case '':
      return 'Idle'
    default:
      return 'Working…'
  }
}

function phaseDot(phase: string): string {
  if (phase === 'COMPLETE') return 'bg-success'
  if (phase === 'ERROR') return 'bg-destructive'
  if (phase === '') return 'bg-muted-foreground/30'
  return 'bg-warning animate-pulse'
}

export function RunFloor({ runsLoading }: { runsLoading: boolean }) {
  const live = useRunStreamStore((s) => s.live)
  const phase = useRunStreamStore((s) => s.phase)
  const level = useRunStreamStore((s) => s.level)
  const totalLevels = useRunStreamStore((s) => s.totalLevels)
  const stations = useRunStreamStore((s) => s.stations)
  const response = useRunStreamStore((s) => s.response)
  const error = useRunStreamStore((s) => s.error)

  const stationList = Object.entries(stations)
  const hasFloor = stationList.length > 0 || live

  // Explicit loading boolean (apps/web/AGENTS.md): skeleton only while the
  // runs fetch is in flight AND nothing has arrived on the floor yet.
  if (runsLoading && !hasFloor) {
    return (
      <div className="space-y-2" aria-busy="true" aria-label="Loading factory floor">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="flex items-center gap-1.5 p-2.5 rounded-lg border border-border/40"
          >
            <Skeleton className="h-6 w-6 rounded shrink-0 bg-muted/20" />
            <div className="flex-1 space-y-1">
              <Skeleton className="h-3.5 w-32 bg-muted/20 rounded-lg" />
              <Skeleton className="h-3 w-20 bg-muted/20 rounded-lg" />
            </div>
            <Skeleton className="h-5 w-16 rounded-full bg-muted/20" />
          </div>
        ))}
      </div>
    )
  }

  if (!hasFloor) {
    return (
      <EmptyCard
        message="No runs on the floor yet — orchestrate a goal to watch it work"
        action={null}
      />
    )
  }

  return (
    <div className="space-y-2" data-testid="run-floor">
      {/* Floor header: phase + liveness, always explicit text */}
      <div className="flex items-center gap-2 flex-wrap">
        <span
          className={cn(
            'inline-block h-2 w-2 rounded-full shrink-0',
            phaseDot(phase),
          )}
          aria-hidden="true"
        />
        <span className="text-[11px] text-muted-foreground">
          {phaseLabel(phase, level, totalLevels)}
        </span>
        <span className="flex-1" />
        <span
          className={cn(
            'text-[11px] px-1.5 py-0.5 rounded-full',
            live ? 'bg-warning/15 text-warning animate-pulse' : 'bg-muted text-muted-foreground',
          )}
        >
          {live ? 'LIVE' : 'IDLE'}
        </span>
      </div>

      {/* Stations: one box per agent task, status as text + dot */}
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {stationList.map(([id, station]) => (
          <div
            key={id}
            className="min-w-0 rounded-lg border border-border/40 px-2.5 py-2 bg-background/40"
            data-testid={`station-${id}`}
          >
            <div className="flex items-center gap-1.5">
              <span
                className={cn(
                  'inline-block h-2 w-2 shrink-0 rounded-full',
                  stationDot(station.status),
                )}
                aria-hidden="true"
              />
              <span className="text-xs font-medium truncate min-w-0 flex-1">
                {station.agent || `task ${id}`}
              </span>
              <span
                className={cn(
                  'text-[11px] px-1.5 py-0.5 rounded-full shrink-0',
                  stationBadge(station.status),
                )}
              >
                {station.status}
              </span>
            </div>
            {station.description && (
              <p className="text-[11px] text-muted-foreground mt-1 line-clamp-2">
                {station.description}
              </p>
            )}
          </div>
        ))}
      </div>

      {/* Composed result */}
      {response && (
        <div className="rounded-lg border border-border/40 bg-muted/30 p-3">
          <p className="text-xs font-medium text-muted-foreground mb-1">Result</p>
          <p className="text-[11px] whitespace-pre-wrap">{response}</p>
        </div>
      )}

      {error && <StatusBanner variant="error" message={error} dismissible={false} />}
    </div>
  )
}
