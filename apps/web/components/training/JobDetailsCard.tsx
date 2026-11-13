'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import { formatElapsed } from '@/lib/formatDuration'
import { formatDateTime } from '@/lib/time-format'
import type { TrainingJob } from '@/lib/training-controller'

interface JobDetailsCardProps {
  job: TrainingJob
}

export function JobDetailsCard({ job }: JobDetailsCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Details</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-xs text-muted-foreground">Job ID</p>
            <p className="font-mono text-xs mt-0.5">{job.id}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Source</p>
            <p className="text-xs mt-0.5">{job.data_source || '—'}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Created</p>
            <p className="text-xs mt-0.5">{formatDateTime(job.created_at) || '—'}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Duration</p>
            <p className="text-xs mt-0.5">{formatElapsed(job.created_at, job.finished_at)}</p>
          </div>
          {job.checkpoint && (
            <div className="col-span-2">
              <p className="text-xs text-muted-foreground">Saved version</p>
              <p className="font-mono text-xs mt-0.5 truncate">{job.checkpoint}</p>
            </div>
          )}
          {job.message && (
            <div className="col-span-2">
              <p className="text-xs text-muted-foreground">Message</p>
              <p className="text-xs mt-0.5 text-muted-foreground">{job.message}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
