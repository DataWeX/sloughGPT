'use client'

import { Card, CardContent, Badge, Button, StatCard, KpiGrid, IconDownload } from '@sloughgpt/strui'
import type { TrainingJob } from '@/lib/training-controller'
import type { JobBadge } from '@/hooks/useTrainingJob'

const RESUMABLE_STATUSES = new Set(['interrupted', 'failed'])

interface JobStatusCardProps {
  job: TrainingJob
  badge: JobBadge | null
  onResume: () => void
  onStop: () => void
  onLoadCheckpoint: () => void
  onDownloadCheckpoint: () => void
  onTryInChat: () => void
}

export function JobStatusCard({
  job,
  badge,
  onResume,
  onStop,
  onLoadCheckpoint,
  onDownloadCheckpoint,
  onTryInChat,
}: JobStatusCardProps) {
  return (
    <Card>
      <CardContent className="pt-4">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <Badge variant={badge?.variant ?? 'outline'}>{badge?.label}</Badge>
            {job.status === 'running' && (
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
              </span>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-1">
            {RESUMABLE_STATUSES.has(job.status) && (
              <Button size="sm" variant="default" className="h-8 text-xs" onClick={onResume}>
                Resume
              </Button>
            )}
            {job.status === 'running' && (
              <Button
                size="sm"
                variant="outline"
                className="h-8 text-xs text-destructive border-destructive/30 hover:bg-destructive/10"
                onClick={onStop}
              >
                Stop
              </Button>
            )}
            {job.checkpoint && job.status === 'completed' && (
              <>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 text-xs"
                  onClick={onLoadCheckpoint}
                >
                  Load saved version
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-8 text-xs"
                  onClick={onDownloadCheckpoint}
                >
                  <IconDownload className="h-4 w-4 mr-1" /> Export
                </Button>
              </>
            )}
            <Button size="sm" variant="outline" className="h-8 text-xs" onClick={onTryInChat}>
              Try in chat
            </Button>
          </div>
        </div>

        {/* Progress bar for running jobs */}
        {job.status === 'running' && (
          <div className="mb-3">
            <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all duration-500 rounded-full"
                style={{ width: `${Math.min(job.progress, 100)}%` }}
              />
            </div>
            <div className="flex items-center justify-between mt-1">
              <p className="text-xs text-muted-foreground">{job.progress}%</p>
              {job.progress > 0 &&
                job.created_at &&
                (() => {
                  const elapsed = (Date.now() - new Date(job.created_at).getTime()) / 1000
                  const rate = job.progress / elapsed
                  const remaining = rate > 0 ? (100 - job.progress) / rate : 0
                  const mins = Math.floor(remaining / 60)
                  const secs = Math.floor(remaining % 60)
                  return (
                    <p className="text-xs text-muted-foreground">
                      ETA: {mins > 0 ? `${mins}m ${secs}s` : `${secs}s`}
                    </p>
                  )
                })()}
            </div>
          </div>
        )}

        <KpiGrid columns={4}>
          <StatCard label="Model" value={job.model || '—'} />
          <StatCard label="Dataset" value={job.dataset || '—'} />
          <StatCard
            label="Epochs"
            value={job.epochs != null ? `${job.current_epoch ?? 0} / ${job.epochs}` : '—'}
          />
          <StatCard label="Steps" value={job.global_step != null ? String(job.global_step) : '—'} />
        </KpiGrid>
        {job.status === 'running' &&
          job.global_step &&
          job.created_at &&
          (() => {
            const elapsed = (Date.now() - new Date(job.created_at).getTime()) / 1000
            const stepsPerMin = elapsed > 0 ? (job.global_step / elapsed) * 60 : 0
            return stepsPerMin > 0 ? (
              <div className="mt-2 flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Speed:</span>
                <span className="text-xs font-mono text-primary">
                  {stepsPerMin.toFixed(1)} steps/min
                </span>
              </div>
            ) : null
          })()}
      </CardContent>
    </Card>
  )
}
