'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import type { WorkflowStatus } from '@/lib/workflow-controller'

interface WorkflowPipelineProps {
  status: WorkflowStatus | null
}

interface PipelineStep {
  key: string
  label: string
  interval: string
  lastRun?: string
  enabled: boolean
}

function buildSteps(status: WorkflowStatus | null): PipelineStep[] {
  if (!status?.config) return []
  const c = status.config
  const s = status.stats
  return [
    {
      key: 'feedback',
      label: 'Feedback',
      interval: 'continuous',
      enabled: status.running,
    },
    {
      key: 'aggregate',
      label: 'Aggregate',
      interval: `${c.aggregate_interval_minutes}m`,
      lastRun: s?.last_aggregate,
      enabled: status.running,
    },
    {
      key: 'train',
      label: 'Train',
      interval: `${c.aggregate_interval_minutes}m`,
      lastRun: s?.last_aggregate,
      enabled: status.running,
    },
    {
      key: 'prune',
      label: 'Prune',
      interval: `${c.prune_interval_minutes}m`,
      lastRun: s?.last_prune,
      enabled: status.running,
    },
    {
      key: 'export',
      label: 'Export',
      interval: `${c.export_interval_hours}h`,
      lastRun: s?.last_export,
      enabled: status.running,
    },
  ]
}

function timeAgo(ts?: string): string {
  if (!ts) return 'never'
  try {
    const d = new Date(ts)
    if (isNaN(d.getTime())) return 'never'
    const diffMs = Date.now() - d.getTime()
    const diffM = Math.floor(diffMs / 60000)
    const diffH = Math.floor(diffM / 60)
    const diffD = Math.floor(diffH / 24)
    if (diffD > 0) return `${diffD}d ago`
    if (diffH > 0) return `${diffH}h ago`
    if (diffM > 0) return `${diffM}m ago`
    return 'just now'
  } catch {
    return 'never'
  }
}

export function WorkflowPipeline({ status }: WorkflowPipelineProps) {
  const steps = buildSteps(status)

  if (steps.length === 0) return null

  return (
    <Card data-testid="workflow-pipeline">
      <CardHeader>
        <CardTitle className="text-base">Pipeline</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-0 overflow-x-auto">
          {steps.map((step, i) => (
            <div key={step.key} className="flex items-center">
              <div className="flex flex-col items-center min-w-[80px]">
                <div className={`h-8 w-8 rounded-full flex items-center justify-center text-xs font-medium ${
                  step.enabled
                    ? 'bg-primary/15 text-primary'
                    : 'bg-muted text-muted-foreground'
                }`}>
                  {i + 1}
                </div>
                <span className="text-[11px] font-medium mt-1.5">{step.label}</span>
                <span className="text-[9px] text-muted-foreground font-mono">{step.interval}</span>
                <span className="text-[9px] text-muted-foreground/60">{timeAgo(step.lastRun)}</span>
              </div>
              {i < steps.length - 1 && (
                <div className={`h-px w-6 mx-0.5 ${step.enabled ? 'bg-primary/30' : 'bg-border'}`} />
              )}
            </div>
          ))}
        </div>
        <div className="mt-3 flex items-center gap-4 text-[10px] text-muted-foreground">
          <div className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-primary/15 border border-primary/30" />
            Active
          </div>
          <div className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-muted border border-border" />
            Idle
          </div>
          <span className="ml-auto">{status?.stats?.feedback_records ?? 0} feedback records</span>
        </div>
      </CardContent>
    </Card>
  )
}
