/**
 * Status presentation helpers and task-rollup derivation for the Agents page.
 */
import type { OrchestrateTask } from '@/lib/agents-controller'

export type RunStatus = 'completed' | 'failed' | 'running' | string

export function statusDotColor(status: RunStatus): string {
  if (status === 'completed') return 'bg-success'
  if (status === 'failed') return 'bg-destructive'
  return 'bg-warning animate-pulse'
}

export function statusTextColor(status: RunStatus): string {
  if (status === 'completed') return 'text-success'
  if (status === 'failed') return 'text-destructive'
  return 'text-warning'
}

export function taskDotColor(status: RunStatus): string {
  if (status === 'completed') return 'bg-success'
  if (status === 'failed') return 'bg-destructive'
  return 'bg-muted-foreground/30'
}

export function taskBadgeStyle(status: RunStatus): string {
  if (status === 'completed') return 'bg-success/15 text-success'
  if (status === 'failed') return 'bg-destructive/15 text-destructive'
  return 'bg-muted text-muted-foreground'
}

export function getAgentTaskStats(
  tasks: OrchestrateTask[],
): Record<string, { completed: number; failed: number; total: number }> {
  const stats: Record<string, { completed: number; failed: number; total: number }> = {}
  for (const t of tasks) {
    if (!stats[t.agent]) stats[t.agent] = { completed: 0, failed: 0, total: 0 }
    stats[t.agent].total++
    if (t.status === 'completed') stats[t.agent].completed++
    else if (t.status === 'failed') stats[t.agent].failed++
  }
  return stats
}
