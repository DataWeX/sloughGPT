'use client'

// Training-job progress card (poll target for the run result's training_job_id).
// Extracted from page.tsx by the page-decomposition refactor.

import { Badge, Button, Card, CardContent, CardHeader, CardTitle, Progress } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import type { VMTrainingJob } from '@/lib/vm-controller'
import { StatusRow } from './VmStatusRow'

export function TrainingCard({ job, onStop }: { job: VMTrainingJob | null; onStop: () => void }) {
  if (!job) return null
  const running = ['running', 'queued', 'starting'].includes(job.status)
  const failed = ['failed', 'cancelled', 'not_found', 'error', 'interrupted'].includes(job.status)
  const pct: number | null = job.progress != null ? Math.round(job.progress * 100) : null
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Training</CardTitle>
          <div className="flex items-center gap-2">
            {running && (
              <Button size="sm" variant="ghost" onClick={onStop}>
                Stop
              </Button>
            )}
            <Badge
              variant={job.status === 'completed' ? 'success' : failed ? 'error' : 'warning'}
              size="sm"
            >
              {job.status}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        <StatusRow label="Job" value={`#${job.job_id}`} />
        {job.api_job_id && <StatusRow label="API job" value={job.api_job_id} />}
        {running && (
          <>
            <Progress
              value={pct}
              variant={pct != null && pct > 0 ? 'default' : 'warning'}
              size="sm"
            />
            <p className="text-xs text-muted-foreground">
              Training in progress — {pct != null ? `${pct}%` : '--'}
            </p>
          </>
        )}
        {job.status === 'completed' && (
          <p className="text-xs text-success">Training completed successfully.</p>
        )}
        {job.status === 'completed' && job.result && (
          <div className="space-y-1">
            <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Result
            </p>
            <pre className="text-xs font-mono bg-muted/30 p-2 rounded overflow-x-auto whitespace-pre-wrap text-success">
              {job.result}
            </pre>
          </div>
        )}
        {job.error && <StatusBanner variant="error" message={job.error} dismissible={false} />}
      </CardContent>
    </Card>
  )
}
