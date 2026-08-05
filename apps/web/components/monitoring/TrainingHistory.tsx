'use client'

import { Card, CardContent } from '@sloughgpt/strui'
import type { TrainingJob } from '@/lib/training-controller'

interface TrainingHistoryProps {
  jobs: TrainingJob[]
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
      status === 'completed' ? 'bg-success/15 text-success' :
      status === 'running' ? 'bg-warning/15 text-warning' :
      status === 'failed' ? 'bg-destructive/15 text-destructive' :
      'bg-muted text-muted-foreground'
    }`}>{status}</span>
  )
}

export function TrainingHistory({ jobs }: TrainingHistoryProps) {
  if (jobs.length === 0) return null

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Training History</span>
      <CardContent className="p-0">
        <div className="space-y-1">
          {jobs.slice(0, 6).map((job) => (
            <div key={job.id} className="flex items-center justify-between text-xs py-1 px-1.5 rounded hover:bg-muted/30 transition-colors">
              <div className="flex items-center gap-1.5 min-w-0">
                <StatusBadge status={job.status} />
                <span className="truncate font-mono text-muted-foreground">{job.name || job.id}</span>
              </div>
              <div className="flex items-center gap-2 text-[10px] text-muted-foreground shrink-0 ml-2 font-mono">
                {job.loss != null && <span>{job.loss.toFixed(3)}</span>}
                {job.epochs_completed != null && <span>ep{job.epochs_completed}</span>}
              </div>
            </div>
          ))}
        </div>
        {jobs.length > 6 && (
          <p className="text-[10px] text-muted-foreground/50 mt-1 font-mono">+{jobs.length - 6} more</p>
        )}
      </CardContent>
    </Card>
  )
}
