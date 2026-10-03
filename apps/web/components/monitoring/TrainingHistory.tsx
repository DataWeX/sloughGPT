'use client'

import { memo, useState, useCallback } from 'react'
import { useRouter } from '@/vite/next-compat/navigation'
import { cn, Card, CardContent } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { trainingJobsController } from '@/lib/training-controller'
import type { TrainingJob } from '@/lib/training-controller'
import { formatToastError } from '@/lib/error-utils'

interface TrainingHistoryProps {
  jobs: TrainingJob[]
  /** Called after a successful resume so the host page can refresh jobs. */
  onRecovered?: (result: {
    status: string
    original_job_id?: string
    recovery_job_id?: string
    checkpoint_path?: string
    message?: string
  }) => void
}

const RESUMABLE = new Set(['interrupted', 'failed'])

function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={cn(
        'text-[10px] px-1.5 py-0.5 rounded font-medium',
        status === 'completed'
          ? 'bg-success/15 text-success'
          : status === 'running'
            ? 'bg-warning/15 text-warning'
            : status === 'failed' || status === 'interrupted'
              ? 'bg-destructive/15 text-destructive'
              : 'bg-muted text-muted-foreground',
      )}
    >
      {status}
    </span>
  )
}

export const TrainingHistory = memo(function TrainingHistory({
  jobs,
  onRecovered,
}: TrainingHistoryProps) {
  const router = useRouter()
  const addToast = useToastStore((s) => s.addToast)
  const [resuming, setResuming] = useState<string | null>(null)

  const handleResume = useCallback(
    async (jobId: string, e: React.MouseEvent) => {
      e.stopPropagation()
      setResuming(jobId)
      try {
        const result = await trainingJobsController.recover(jobId)
        addToast(result.message || 'Resume started', 'success')
        onRecovered?.(result)
      } catch (e) {
        addToast(formatToastError(e, 'Could not resume job'), 'error')
      } finally {
        setResuming(null)
      }
    },
    [addToast, onRecovered],
  )

  return (
    <Card className="p-2.5">
      <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 block">
        Training History
      </span>
      <CardContent className="p-0">
        {jobs.length === 0 ? (
          <div className="text-[10px] text-muted-foreground/60 text-center py-3 space-y-1.5">
            <div>No training jobs yet</div>
            <Button
              size="sm"
              variant="outline"
              className="h-6 text-[10px]"
              onClick={() => router.push('/training')}
            >
              Go to Training
            </Button>
          </div>
        ) : (
          <div className="space-y-px">
            {jobs.slice(0, 6).map((job) => (
              <div
                key={job.id}
                role="button"
                tabIndex={0}
                onClick={() => router.push(`/training/job/${job.id}`)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    router.push(`/training/job/${job.id}`)
                  }
                }}
                className="w-full flex items-center justify-between text-[10px] py-0.5 px-1 rounded hover:bg-muted/20 transition-colors text-left cursor-pointer"
              >
                <div className="flex items-center gap-1 min-w-0">
                  <StatusBadge status={job.status} />
                  <span className="truncate font-mono text-muted-foreground/60">
                    {job.name || job.id}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground/60 shrink-0 ml-1.5 font-mono tabular-nums">
                  {RESUMABLE.has(job.status) && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-4 text-[10px] px-1"
                      disabled={resuming === job.id}
                      onClick={(e) => {
                        e.stopPropagation()
                        void handleResume(job.id, e)
                      }}
                    >
                      {resuming === job.id ? '…' : 'Resume'}
                    </Button>
                  )}
                  {job.loss != null && <span>{job.loss.toFixed(3)}</span>}
                  {job.epochs_completed != null && <span>ep{job.epochs_completed}</span>}
                </div>
              </div>
            ))}
          </div>
        )}
        {jobs.length > 6 && (
          <p className="text-[10px] text-muted-foreground/40 mt-1 font-mono tabular-nums">
            +{jobs.length - 6} more
          </p>
        )}
      </CardContent>
    </Card>
  )
})
