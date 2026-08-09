'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import type { WorkflowStatus } from '@/lib/workflow-controller'

interface WorkflowHealthCardProps {
  status: WorkflowStatus | null
}

function timeSince(iso?: string): string {
  if (!iso) return 'never'
  try {
    const diff = Date.now() - new Date(iso).getTime()
    const mins = Math.floor(diff / 60000)
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
  const fbPerAdapter = (stats.adapters_count ?? 0) > 0
    ? ((stats.feedback_records ?? 0) / (stats.adapters_count ?? 1)).toFixed(1)
    : '—'

  const lastOps = [
    { label: 'Aggregate', time: stats.last_aggregate },
    { label: 'Prune', time: stats.last_prune },
    { label: 'Export', time: stats.last_export },
  ]

  const staleOps = lastOps.filter(op => {
    if (!op.time) return true
    const diff = Date.now() - new Date(op.time).getTime()
    return diff > 3600000
  }).length

  const healthStatus = !running ? 'Stopped' : staleOps === 0 ? 'Healthy' : staleOps < 3 ? 'Degraded' : 'Stale'
  const healthColor = healthStatus === 'Healthy' ? 'text-success' : healthStatus === 'Degraded' ? 'text-warning' : healthStatus === 'Stopped' ? 'text-muted-foreground' : 'text-destructive'

  return (
    <Card data-testid="workflow-health">
      <CardHeader>
        <CardTitle className="text-base">Workflow Health</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-2 mb-3">
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Status</div>
            <div className={`text-sm font-mono font-medium ${healthColor}`}>{healthStatus}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Feedback</div>
            <div className="text-sm font-mono font-medium">{stats.feedback_records ?? 0}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Fb/Adapter</div>
            <div className="text-sm font-mono font-medium">{fbPerAdapter}</div>
          </div>
        </div>

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
