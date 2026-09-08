'use client'

import { cn, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import type { WorkflowStatus } from '@/lib/workflow-controller'
import { timeAgo } from '@/lib/time-ago'

interface WorkflowPipelineProps {
  status: WorkflowStatus | null
}

interface PipelineStep {
  key: string
  label: string
  interval: string
  lastRun?: number
  enabled: boolean
}

function buildSteps(status: WorkflowStatus | null): PipelineStep[] {
  if (!status?.config) return []
  const c = status.config
  const lr = status.last_runs
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
      lastRun: lr?.aggregate,
      enabled: status.running,
    },
    {
      key: 'train',
      label: 'Train',
      interval: `${c.aggregate_interval_minutes}m`,
      lastRun: lr?.aggregate,
      enabled: status.running,
    },
    {
      key: 'prune',
      label: 'Prune',
      interval: `${c.prune_interval_minutes}m`,
      lastRun: lr?.prune,
      enabled: status.running,
    },
    {
      key: 'export',
      label: 'Export',
      interval: `${c.export_interval_hours}h`,
      lastRun: lr?.export,
      enabled: status.running,
    },
  ]
}

export function WorkflowPipeline({ status }: WorkflowPipelineProps) {
  const steps = buildSteps(status)

  if (status === null) {
    return (
      <Card data-testid="workflow-pipeline">
        <CardContent className="py-5">
          <p className="text-[11px] text-muted-foreground/60 text-center">Pipeline not configured</p>
        </CardContent>
      </Card>
    )
  }

  if (steps.length === 0) return null

  return (
    <Card data-testid="workflow-pipeline">
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Pipeline</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-0 overflow-x-auto">
          {steps.map((step, i) => (
            <div key={step.key} className="flex items-center">
              <div className="flex flex-col items-center min-w-[72px]">
                <div className={cn('h-7 w-7 rounded-full flex items-center justify-center text-[10px] font-medium', step.enabled
                    ? 'bg-primary/15 text-primary'
                    : 'bg-muted text-muted-foreground')}>
                  {i + 1}
                </div>
                <span className="text-[10px] font-medium mt-1">{step.label}</span>
                <span className="text-[9px] text-muted-foreground/60 font-mono tabular-nums">{step.interval}</span>
                <span className="text-[9px] text-muted-foreground/50">{timeAgo(step.lastRun)}</span>
              </div>
              {i < steps.length - 1 && (
                <div className={cn('h-px w-5 mx-0.5', step.enabled ? 'bg-primary/30' : 'bg-border')} />
              )}
            </div>
          ))}
        </div>
        <div className="mt-2.5 flex items-center gap-3 text-[9px] text-muted-foreground/60">
          <div className="flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-primary/15 border border-primary/30" />
            Active
          </div>
          <div className="flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-muted border border-border" />
            Idle
          </div>
          <span className="ml-auto tabular-nums">{status?.stats?.feedback_recorded ?? 0} feedback records</span>
        </div>
      </CardContent>
    </Card>
  )
}
