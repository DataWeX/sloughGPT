'use client'

import { Card, CardContent, CardHeader, CardTitle, Badge, Skeleton } from '@sloughgpt/strui'

export interface CloudJob {
  job_id: string
  provider: string
  status: string
  progress: number
  error?: string
}

interface CloudTrainingJobListProps {
  jobs?: CloudJob[]
  loading?: boolean
}

function statusVariant(status: string) {
  if (status === 'completed') return 'default'
  if (status === 'failed') return 'destructive'
  return 'secondary'
}

export function CloudTrainingJobList({
  jobs = [],
  loading = false,
}: CloudTrainingJobListProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Training Jobs</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-20 w-full" />
        ) : jobs.length === 0 ? (
          <p className="text-sm text-muted-foreground">No training jobs yet.</p>
        ) : (
          <div className="space-y-2">
            {jobs.map((job) => (
              <div
                key={job.job_id}
                className="flex items-center justify-between p-2 border border-border/50 rounded"
              >
                <div>
                  <span className="font-mono text-xs">{job.job_id}</span>
                  <span className="ml-2 text-xs text-muted-foreground">
                    {job.provider}
                  </span>
                </div>
                <Badge variant={statusVariant(job.status) as any}>
                  {job.status}
                </Badge>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
