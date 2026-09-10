'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, EmptyCard, Skeleton, StatusBanner, cn } from '@sloughgpt/strui'
import { IconRefresh } from '@/components/icons/NavIcons'

interface AgentRunTask {
  id: string
  agent: string
  description: string
  status: string
  result_preview?: string
  depends_on?: string[]
}

interface AgentRun {
  id: string
  status: string
  goal: string
  response?: string
  error?: string
  tasks: AgentRunTask[]
  completed_count: number
  failed_count: number
  started_at?: string
  finished_at?: string
  logs: string[]
}

interface AgentRunsHistoryProps {
  runs: AgentRun[]
  loading: boolean
  viewMode: 'list' | 'timeline'
  statusFilter: string | null
  agentFilter: string | null
  expandedRun: string | null
  onRefresh: () => void
  onViewModeChange: (mode: 'list' | 'timeline') => void
  onStatusFilterChange: (status: string | null) => void
  onAgentFilterChange: (agent: string | null) => void
  onExpandRun: (id: string | null) => void
}

function statusDotColor(status: string): string {
  if (status === 'completed') return 'bg-success'
  if (status === 'failed') return 'bg-destructive'
  return 'bg-warning animate-pulse'
}

function statusTextColor(status: string): string {
  if (status === 'completed') return 'text-success'
  if (status === 'failed') return 'text-destructive'
  return 'text-warning'
}

function taskDotColor(status: string): string {
  if (status === 'completed') return 'bg-success'
  if (status === 'failed') return 'bg-destructive'
  return 'bg-muted-foreground/30'
}

function taskBadgeStyle(status: string): string {
  if (status === 'completed') return 'bg-success/15 text-success'
  if (status === 'failed') return 'bg-destructive/15 text-destructive'
  return 'bg-muted text-muted-foreground'
}

function getAgentTaskStats(tasks: AgentRunTask[]): Record<string, { completed: number; failed: number; total: number }> {
  const stats: Record<string, { completed: number; failed: number; total: number }> = {}
  for (const t of tasks) {
    if (!stats[t.agent]) stats[t.agent] = { completed: 0, failed: 0, total: 0 }
    stats[t.agent].total++
    if (t.status === 'completed') stats[t.agent].completed++
    else if (t.status === 'failed') stats[t.agent].failed++
  }
  return stats
}

function formatElapsed(started: string, finished?: string): string {
  const s = new Date(started).getTime()
  const e = finished ? new Date(finished).getTime() : Date.now()
  const ms = e - s
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}m`
}

export function AgentRunsHistory({
  runs,
  loading,
  viewMode,
  statusFilter,
  agentFilter,
  expandedRun,
  onRefresh,
  onViewModeChange,
  onStatusFilterChange,
  onAgentFilterChange,
  onExpandRun,
}: AgentRunsHistoryProps) {
  const filtered = runs
    .filter(r => !statusFilter || r.status === statusFilter)
    .filter(r => !agentFilter || r.tasks.some(t => t.agent === agentFilter))

  const agentNames = new Set<string>()
  runs.forEach(r => r.tasks.forEach(t => { if (t.agent) agentNames.add(t.agent) }))

  return (
    <Card data-testid="agent-runs-history">
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <div className="flex items-center justify-between">
          <CardTitle className="text-[11px] font-medium">Run History</CardTitle>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => onViewModeChange('list')}
              className={cn('text-[10px] px-2 py-1 rounded transition-colors', viewMode === 'list' ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:bg-muted/80')}
            >
              List
            </button>
            <button
              type="button"
              onClick={() => onViewModeChange('timeline')}
              className={cn('text-[10px] px-2 py-1 rounded transition-colors', viewMode === 'timeline' ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:bg-muted/80')}
            >
              Timeline
            </button>
            <Button size="sm" variant="ghost" onClick={onRefresh} disabled={loading} className="h-6 text-[10px]">
              <IconRefresh className="h-4 w-4 mr-1" />
              Refresh
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5 space-y-2">
        {loading ? (
          <div className="space-y-2" data-testid="runs-loading">
            {[1, 2, 3].map(i => (
              <div key={i} className="flex items-center gap-1.5 p-2.5 rounded-lg border border-border/40">
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
          <EmptyCard message="No runs yet — orchestrate a goal to see history here" action={null} />
        ) : (
          <div className="space-y-3">
            {runs.length > 2 && (
              <div className="space-y-2 pb-2" data-testid="runs-filters">
                <div className="flex items-center gap-1.5">
                  <div className="flex gap-1">
                    {[null, 'completed', 'failed', 'running'].map(s => (
                      <button
                        key={s ?? 'all'}
                        type="button"
                        onClick={() => onStatusFilterChange(s)}
                        className={cn('text-[10px] px-2 py-1 rounded-full border transition-colors', statusFilter === s ? 'bg-primary/15 text-primary border-primary/30' : 'border-border/40 text-muted-foreground hover:bg-muted/80')}
                      >
                        {s === null ? 'All' : s}
                      </button>
                    ))}
                  </div>
                  <div className="flex-1" />
                  <div className="flex gap-1.5 text-[10px] text-muted-foreground/60">
                    <span>{runs.filter(r => r.status === 'completed').length} completed</span>
                    <span>{runs.filter(r => r.status === 'failed').length} failed</span>
                  </div>
                </div>
                {agentNames.size > 1 && (
                  <div className="flex flex-wrap gap-1" data-testid="agent-filters">
                    <button
                      type="button"
                      onClick={() => onAgentFilterChange(null)}
                      className={cn('text-[10px] px-2 py-1 rounded border transition-colors', agentFilter === null ? 'bg-primary/15 text-primary border-primary/30' : 'border-border/40 text-muted-foreground hover:bg-muted/80')}
                    >
                      All agents
                    </button>
                    {Array.from(agentNames).sort().map(name => (
                      <button
                        key={name}
                        type="button"
                        onClick={() => onAgentFilterChange(agentFilter === name ? null : name)}
                        className={cn('text-[10px] px-2 py-1 rounded border transition-colors', agentFilter === name ? 'bg-primary/15 text-primary border-primary/30' : 'border-border/40 text-muted-foreground hover:bg-muted/80')}
                      >
                        {name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            {viewMode === 'list' ? (
              <div data-testid="runs-list">
                {filtered.map(run => {
                  const expanded = expandedRun === run.id
                  return (
                    <div key={run.id} className={cn('rounded-lg border border-border/40 transition-colors mb-2', expanded ? 'border-primary/40' : 'hover:bg-muted/20')}>
                      <button
                        type="button"
                        className="w-full flex items-center gap-2 px-3 py-2 text-left"
                        onClick={() => onExpandRun(expanded ? null : run.id)}
                        aria-expanded={expanded}
                        data-testid={`run-${run.id}`}
                      >
                        <span className={cn('inline-block h-2 w-2 shrink-0 rounded-full', statusDotColor(run.status))} />
                        <span className="flex-1 truncate text-[11px]">{run.goal}</span>
                        <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                          {run.completed_count}/{run.completed_count + run.failed_count} tasks
                        </span>
                      </button>
                      {expanded && (
                        <div className="border-t border-border/40 px-3 py-2 space-y-3">
                          {run.error && (
                            <StatusBanner variant="error" message={run.error} dismissible={false} />
                          )}
                          {run.tasks.length > 0 && (
                            <div className="space-y-1.5">
                              <p className="text-[10px] font-medium text-muted-foreground">Tasks ({run.tasks.length})</p>
                              {run.tasks.map(task => (
                                <div key={task.id} className="flex items-center gap-2 rounded-lg border border-border/40 px-2.5 py-2 text-[11px]">
                                  <span className={cn('inline-block h-2 w-2 shrink-0 rounded-full', task.status === 'completed' ? 'bg-success' : task.status === 'failed' ? 'bg-destructive' : 'bg-muted-foreground/30')} />
                                  <span className="font-medium text-[10px] min-w-[64px] text-muted-foreground">{task.agent}</span>
                                  <span className="flex-1 truncate">{task.description}</span>
                                </div>
                              ))}
                            </div>
                          )}
                          {run.response && (
                            <div className="rounded-lg border border-border/40 bg-muted/30 p-3">
                              <p className="text-[10px] font-medium text-muted-foreground mb-1">Result</p>
                              <p className="text-[11px] whitespace-pre-wrap line-clamp-4">{run.response}</p>
                            </div>
                          )}
                          {run.logs.length > 0 && (
                            <div className="rounded-lg bg-muted/20 p-3">
                              <p className="text-[10px] font-medium text-muted-foreground mb-1">Logs ({run.logs.length})</p>
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
              </div>
            ) : (
              <div className="space-y-4" data-testid="runs-timeline">
                {filtered.map(run => (
                  <div key={run.id} className="space-y-2">
                    <div className="flex items-center gap-2">
                      <span className={cn('text-[10px] font-medium rounded-full', statusTextColor(run.status))}>{run.status}</span>
                      <span className="text-[11px] flex-1 truncate">{run.goal}</span>
                    </div>
                    <div className="ml-2 border-l-2 border-border/40 pl-4 space-y-2">
                      {run.tasks.map((task, i) => (
                        <div key={task.id} className="relative flex items-center gap-1.5">
                          <div className={cn('absolute -left-[21px] h-2.5 w-2.5 rounded-full border-2 border-background', taskDotColor(task.status))} />
                          <span className="text-[10px] text-muted-foreground w-8 shrink-0">#{i + 1}</span>
                          <span className="text-[10px] text-muted-foreground min-w-[64px] shrink-0">{task.agent}</span>
                          <span className="text-[10px] flex-1 truncate">{task.description}</span>
                          <span className={cn('text-[10px] px-1.5 py-0.5 rounded-full', taskBadgeStyle(task.status))}>
                            {task.status}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
