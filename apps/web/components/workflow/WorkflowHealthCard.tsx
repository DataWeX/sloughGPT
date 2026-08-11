'use client'

import { Card, CardHeader, CardTitle, CardContent, StatCard, KpiGrid } from '@sloughgpt/strui'
import type { WorkflowStatus } from '@/lib/workflow-controller'

interface WorkflowHealthCardProps {
  status: WorkflowStatus | null
}

function timeSince(ts?: number): string {
  if (!ts || ts === 0) return 'never'
  try {
    const diff = Date.now() / 1000 - ts
    const mins = Math.floor(diff / 60)
    const hrs = Math.floor(mins / 60)
    const days = Math.floor(hrs / 24)
    if (days > 0) return `${days}d ago`
    if (hrs > 0) return `${hrs}h ago`
    if (mins > 0) return `${mins}m ago`
    return 'just now'
  } catch { return 'unknown' }
}

export function WorkflowHealthCard({ status }: WorkflowHealthCardProps) {
  if (!status?.stats) return null

  const { stats, running } = status
  const fbPerAdapter = (stats.user_adapter_trained ?? 0) > 0
    ? ((stats.feedback_recorded ?? 0) / (stats.user_adapter_trained ?? 1)).toFixed(1)
    : '—'

  const lastOps = [
    { label: 'Aggregate', time: status.last_runs?.aggregate },
    { label: 'Prune', time: status.last_runs?.prune },
    { label: 'Export', time: status.last_runs?.export },
  ]

  const staleOps = lastOps.filter(op => {
    if (!op.time) return true
    const diff = Date.now() / 1000 - op.time
    return diff > 3600
  }).length

  const healthStatus = !running ? 'Stopped' : staleOps === 0 ? 'Healthy' : staleOps < 3 ? 'Degraded' : 'Stale'
  const healthColor = healthStatus === 'Healthy' ? 'text-success' : healthStatus === 'Degraded' ? 'text-warning' : healthStatus === 'Stopped' ? 'text-muted-foreground' : 'text-destructive'

  return (
    <Card data-testid="workflow-health">
      <CardHeader>
        <CardTitle className="text-base">Workflow Health</CardTitle>
      </CardHeader>
      <CardContent>
        <KpiGrid columns={3} className="mb-3">
          <StatCard label="Status" value={<span className={healthColor}>{healthStatus}</span>} />
          <StatCard label="Feedback" value={stats.feedback_recorded ?? 0} />
          <StatCard label="Fb/Trained" value={fbPerAdapter} />
        </KpiGrid>

        <div className="space-y-1.5">
          <div className="text-[10px] text-muted-foreground mb-1">Last Operations</div>
          {lastOps.map(op => (
            <div key={op.label} className="flex items-center justify-between text-[11px] py-0.5 border-b border-border/30 last:border-0">
              <span className="text-muted-foreground">{op.label}</span>
              <span className={`font-mono ${!op.time ? 'text-warning' : ''}`}>{timeSince(op.time)}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
