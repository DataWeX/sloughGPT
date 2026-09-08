'use client'

import { memo } from 'react'
import { useRouter } from 'next/navigation'
import { cn, Card, CardContent } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import type { TrainingJob } from '@/lib/training-controller'

interface TrainingHistoryProps {
  jobs: TrainingJob[]
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span className={cn('text-[9px] px-1.5 py-0.5 rounded font-medium', status === 'completed' ? 'bg-success/15 text-success' :
      status === 'running' ? 'bg-warning/15 text-warning' :
      status === 'failed' ? 'bg-destructive/15 text-destructive' :
      'bg-muted text-muted-foreground')}>{status}</span>
  )
}

export const TrainingHistory = memo(function TrainingHistory({ jobs }: TrainingHistoryProps) {
  const router = useRouter()
  return (
    <Card className="p-2.5">
      <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 block">Training History</span>
      <CardContent className="p-0">
        {jobs.length === 0 ? (
          <div className="text-[10px] text-muted-foreground/60 text-center py-3 space-y-1.5">
            <div>No training jobs yet</div>
            <Button size="sm" variant="outline" className="h-6 text-[9px]" onClick={() => router.push('/training')}>
              Go to Training
            </Button>
          </div>
        ) : (
          <div className="space-y-px">
            {jobs.slice(0, 6).map((job) => (
              <div key={job.id} className="flex items-center justify-between text-[10px] py-0.5 px-1 rounded hover:bg-muted/20 transition-colors">
                <div className="flex items-center gap-1 min-w-0">
                  <StatusBadge status={job.status} />
                  <span className="truncate font-mono text-muted-foreground/60">{job.name || job.id}</span>
                </div>
                <div className="flex items-center gap-1.5 text-[9px] text-muted-foreground/60 shrink-0 ml-1.5 font-mono tabular-nums">
                  {job.loss != null && <span>{job.loss.toFixed(3)}</span>}
                  {job.epochs_completed != null && <span>ep{job.epochs_completed}</span>}
                </div>
              </div>
            ))}
          </div>
        )}
        {jobs.length > 6 && (
          <p className="text-[9px] text-muted-foreground/40 mt-1 font-mono tabular-nums">+{jobs.length - 6} more</p>
        )}
      </CardContent>
    </Card>
  )
})
