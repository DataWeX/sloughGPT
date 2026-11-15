'use client'

/**
 * Run history with status/agent filters, list/timeline views and expansion.
 */
import type { AgentsWorkbench } from './useAgentsWorkbench'
import {
  statusDotColor,
  statusTextColor,
  taskDotColor,
  taskBadgeStyle,
  getAgentTaskStats,
} from './agents-format'
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  EmptyCard,
  IconRefresh,
  Skeleton,
  cn,
} from '@sloughgpt/strui'
import { SectionErrorBoundary } from '@/components/SectionErrorBoundary'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { formatElapsed } from '@/lib/formatDuration'

export function RunHistoryCard({
  runs,
  runsLoading,
  expandedRun,
  setExpandedRun,
  runStatusFilter,
  setRunStatusFilter,
  runAgentFilter,
  setRunAgentFilter,
  runViewMode,
  setRunViewMode,
  fetchRuns,
}: {
  runs: AgentsWorkbench['runs']
  runsLoading: AgentsWorkbench['runsLoading']
  expandedRun: AgentsWorkbench['expandedRun']
  setExpandedRun: AgentsWorkbench['setExpandedRun']
  runStatusFilter: AgentsWorkbench['runStatusFilter']
  setRunStatusFilter: AgentsWorkbench['setRunStatusFilter']
  runAgentFilter: AgentsWorkbench['runAgentFilter']
  setRunAgentFilter: AgentsWorkbench['setRunAgentFilter']
  runViewMode: AgentsWorkbench['runViewMode']
  setRunViewMode: AgentsWorkbench['setRunViewMode']
  fetchRuns: AgentsWorkbench['fetchRuns']
}) {
  return (
    <SectionErrorBoundary sectionName="Run history viewer">
      <Card>
        <CardHeader className="pb-2 pt-2.5 px-2.5">
          <div className="flex items-center justify-between">
            <CardTitle className="text-[11px] font-medium">Run History</CardTitle>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => setRunViewMode('list')}
                className={cn(
                  'text-[10px] px-2 py-1 rounded transition-colors',
                  runViewMode === 'list'
                    ? 'bg-primary/15 text-primary'
                    : 'text-muted-foreground hover:bg-muted/80',
                )}
              >
                List
              </button>
              <button
                type="button"
                onClick={() => setRunViewMode('timeline')}
                className={cn(
                  'text-[10px] px-2 py-1 rounded transition-colors',
                  runViewMode === 'timeline'
                    ? 'bg-primary/15 text-primary'
                    : 'text-muted-foreground hover:bg-muted/80',
                )}
              >
                Timeline
              </button>
              <Button
                size="sm"
                variant="ghost"
                onClick={fetchRuns}
                disabled={runsLoading}
                className="h-6 text-[10px]"
              >
                <IconRefresh className="h-4 w-4 mr-1" />
                Refresh
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="px-2.5 pb-2.5 space-y-2">
          {runsLoading ? (
            <div className="space-y-2">
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
          ) : runs.length === 0 ? (
            <EmptyCard
              message="No runs yet — orchestrate a goal to see history here"
              action={null}
            />
          ) : (
            <div className="space-y-3">
              {runs.length > 2 && (
                <div className="space-y-2 pb-2">
                  <div className="flex items-center gap-1.5">
                    <div className="flex gap-1">
                      {[null, 'completed', 'failed', 'running'].map((s) => (
                        <button
                          key={s ?? 'all'}
                          type="button"
                          onClick={() => setRunStatusFilter(s)}
                          className={cn(
                            'text-[10px] px-2 py-1 rounded-full border transition-colors',
                            runStatusFilter === s
                              ? 'bg-primary/15 text-primary border-primary/30'
                              : 'border-border/40 text-muted-foreground hover:bg-muted/80',
                          )}
                        >
                          {s === null ? 'All' : s}
                        </button>
                      ))}
                    </div>
                    <div className="flex-1" />
                    <div className="flex gap-1.5 text-[10px] text-muted-foreground/60">
                      <span>{runs.filter((r) => r.status === 'completed').length} completed</span>
                      <span>{runs.filter((r) => r.status === 'failed').length} failed</span>
                      <span>
                        {runs.reduce((s, r) => s + r.completed_count + r.failed_count, 0)} total
                        tasks
                      </span>
                    </div>
                  </div>
                  {(() => {
                    const agentNames = new Set<string>()
                    runs.forEach((r) =>
                      r.tasks.forEach((t) => {
                        if (t.agent) agentNames.add(t.agent)
                      }),
                    )
                    if (agentNames.size <= 1) return null
                    return (
                      <div className="flex flex-wrap gap-1">
                        <button
                          type="button"
                          onClick={() => setRunAgentFilter(null)}
                          className={cn(
                            'text-[10px] px-2 py-1 rounded border transition-colors',
                            runAgentFilter === null
                              ? 'bg-primary/15 text-primary border-primary/30'
                              : 'border-border/40 text-muted-foreground hover:bg-muted/80',
                          )}
                        >
                          All agents
                        </button>
                        {Array.from(agentNames)
                          .sort()
                          .map((name) => (
                            <button
                              key={name}
                              type="button"
                              onClick={() =>
                                setRunAgentFilter(runAgentFilter === name ? null : name)
                              }
                              className={cn(
                                'text-[10px] px-2 py-1 rounded border transition-colors',
                                runAgentFilter === name
                                  ? 'bg-primary/15 text-primary border-primary/30'
                                  : 'border-border/40 text-muted-foreground hover:bg-muted/80',
                              )}
                            >
                              {name}
                            </button>
                          ))}
                      </div>
                    )
                  })()}
                </div>
              )}
              {runViewMode === 'list' ? (
                <>
                  {runs
                    .filter((r) => !runStatusFilter || r.status === runStatusFilter)
                    .filter(
                      (r) => !runAgentFilter || r.tasks.some((t) => t.agent === runAgentFilter),
                    )
                    .map((run) => {
                      const expanded = expandedRun === run.id
                      const statusColor = statusDotColor(run.status)
                      return (
                        <div
                          key={run.id}
                          className={cn(
                            'rounded-lg border border-border/40 transition-colors',
                            expanded ? 'border-primary/40' : 'hover:bg-muted/20',
                          )}
                        >
                          <button
                            type="button"
                            className="w-full flex items-center gap-2 px-3 py-2 text-left"
                            onClick={() => setExpandedRun(expanded ? null : run.id)}
                            aria-expanded={expanded}
                          >
                            <span
                              className={cn(
                                'inline-block h-2 w-2 shrink-0 rounded-full',
                                statusColor,
                              )}
                            />
                            <span className="flex-1 truncate text-[11px]">{run.goal}</span>
                            <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                              {run.completed_count}/{run.completed_count + run.failed_count} tasks
                            </span>
                            {run.started_at && (
                              <span className="shrink-0 text-[10px] text-muted-foreground/70">
                                {formatElapsed(run.started_at, run.finished_at)}
                              </span>
                            )}
                            <span className="shrink-0 text-[10px] text-muted-foreground">
                              {run.started_at ? new Date(run.started_at).toLocaleString() : ''}
                            </span>
                          </button>
                          {expanded && (
                            <div className="border-t border-border/40 px-3 py-2 space-y-3">
                              {run.error && (
                                <StatusBanner
                                  variant="error"
                                  message={run.error}
                                  dismissible={false}
                                />
                              )}
                              {run.tasks.length > 0 &&
                                (() => {
                                  const agentStats = getAgentTaskStats(run.tasks)
                                  const agentNames = Object.keys(agentStats)
                                  return (
                                    <div className="space-y-2">
                                      {agentNames.length > 1 && (
                                        <div className="flex flex-wrap gap-2">
                                          {agentNames.map((name) => {
                                            const s = agentStats[name]
                                            return (
                                              <span
                                                key={name}
                                                className="text-[10px] px-2 py-1 rounded bg-muted text-muted-foreground font-medium"
                                              >
                                                {name}: {s.completed}/{s.total}
                                                {s.failed > 0 && (
                                                  <span className="text-destructive ml-1">
                                                    ({s.failed} failed)
                                                  </span>
                                                )}
                                              </span>
                                            )
                                          })}
                                        </div>
                                      )}
                                      <div className="space-y-1.5">
                                        <p className="text-[10px] font-medium text-muted-foreground">
                                          Tasks ({run.tasks.length})
                                        </p>
                                        {run.tasks.map((task) => (
                                          <div
                                            key={task.id}
                                            className="flex items-center gap-2 rounded-lg border border-border/40 px-2.5 py-2 text-[11px]"
                                          >
                                            <span
                                              className={cn(
                                                'inline-block h-2 w-2 shrink-0 rounded-full',
                                                task.status === 'completed'
                                                  ? 'bg-success'
                                                  : task.status === 'failed'
                                                    ? 'bg-destructive'
                                                    : 'bg-muted-foreground/30',
                                              )}
                                            />
                                            <span className="font-medium text-[10px] min-w-[64px] text-muted-foreground">
                                              {task.agent}
                                            </span>
                                            <span className="flex-1 truncate">
                                              {task.description}
                                            </span>
                                            {task.result_preview && (
                                              <span className="text-[10px] text-muted-foreground/50 truncate max-w-[200px]">
                                                {task.result_preview}
                                              </span>
                                            )}
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )
                                })()}
                              {run.response && (
                                <div className="rounded-lg border border-border/40 bg-muted/30 p-3">
                                  <p className="text-[10px] font-medium text-muted-foreground mb-1">
                                    Result
                                  </p>
                                  <p className="text-[11px] whitespace-pre-wrap line-clamp-4">
                                    {run.response}
                                  </p>
                                </div>
                              )}
                              {run.logs.length > 0 && (
                                <div className="rounded-lg bg-muted/20 p-3">
                                  <p className="text-[10px] font-medium text-muted-foreground mb-1">
                                    Logs ({run.logs.length})
                                  </p>
                                  <pre className="text-[10px] text-muted-foreground whitespace-pre-wrap max-h-40 overflow-y-auto font-mono">
                                    {run.logs.join('\n')}
                                  </pre>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )
                    })}
                </>
              ) : (
                <div className="space-y-4">
                  {runs
                    .filter((r) => !runStatusFilter || r.status === runStatusFilter)
                    .filter(
                      (r) => !runAgentFilter || r.tasks.some((t) => t.agent === runAgentFilter),
                    )
                    .map((run) => {
                      const statusColor = statusTextColor(run.status)
                      return (
                        <div key={run.id} className="space-y-2">
                          <div className="flex items-center gap-2">
                            <span
                              className={cn('text-[10px] font-medium rounded-full', statusColor)}
                            >
                              {run.status}
                            </span>
                            <span className="text-[11px] flex-1 truncate">{run.goal}</span>
                            {run.started_at && (
                              <span className="text-[10px] text-muted-foreground/70">
                                {formatElapsed(run.started_at, run.finished_at)}
                              </span>
                            )}
                            <span className="text-[10px] text-muted-foreground">
                              {run.started_at ? new Date(run.started_at).toLocaleString() : ''}
                            </span>
                          </div>
                          <div className="ml-2 border-l-2 border-border/40 pl-4 space-y-2">
                            {run.tasks.map((task, i) => {
                              const taskColor = taskDotColor(task.status)
                              return (
                                <div key={task.id} className="relative flex items-center gap-1.5">
                                  <div
                                    className={cn(
                                      'absolute -left-[21px] h-2.5 w-2.5 rounded-full border-2 border-background',
                                      taskColor,
                                    )}
                                  />
                                  <span className="text-[10px] text-muted-foreground w-8 shrink-0">
                                    #{i + 1}
                                  </span>
                                  <span className="text-[10px] text-muted-foreground min-w-[64px] shrink-0">
                                    {task.agent}
                                  </span>
                                  <span className="text-[10px] flex-1 truncate">
                                    {task.description}
                                  </span>
                                  <span
                                    className={cn(
                                      'text-[10px] px-1.5 py-0.5 rounded-full',
                                      taskBadgeStyle(task.status),
                                    )}
                                  >
                                    {task.status}
                                  </span>
                                </div>
                              )
                            })}
                            {run.tasks.length === 0 && (
                              <p className="text-[10px] text-muted-foreground/60 italic">
                                No tasks recorded
                              </p>
                            )}
                          </div>
                        </div>
                      )
                    })}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </SectionErrorBoundary>
  )
}
