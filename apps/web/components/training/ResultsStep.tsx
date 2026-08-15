'use client'

import { useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Badge } from '@sloughgpt/strui'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'
import type { TrainingJob } from '@/lib/training-controller'

interface ResultsStepProps {
  checkpoints: UseTrainingCheckpointsReturn
  goToTrain: () => void
  onTest: () => void
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}

function jobStatusStyle(status: string): { label: string; variant: 'success' | 'warning' | 'error' | 'secondary' } {
  const s = status.toLowerCase()
  if (s === 'running' || s === 'training' || s === 'queued' || s === 'pending') {
    return { label: s === 'running' ? 'Running' : status, variant: 'warning' }
  }
  if (s === 'completed' || s === 'complete' || s === 'success' || s === 'done') {
    return { label: 'Completed', variant: 'success' }
  }
  if (s === 'failed' || s === 'error' || s === 'cancelled' || s === 'canceled') {
    return { label: s === 'failed' ? 'Failed' : status, variant: 'error' }
  }
  return { label: status, variant: 'secondary' }
}

function formatJobLine(job: TrainingJob): string {
  const parts: string[] = []
  if (job.method) parts.push(job.method)
  if (job.dataset) parts.push(job.dataset)
  return parts.join(' · ')
}

export function ResultsStep({ checkpoints, goToTrain, onTest, addToast }: ResultsStepProps) {
  const bestName = useMemo(() => {
    const withLoss = checkpoints.checkpoints.filter(c => c.loss != null && c.loss > 0)
    if (withLoss.length === 0) return null
    return withLoss.reduce((min, c) => (c.loss! < min.loss! ? c : min), withLoss[0]).name
  }, [checkpoints.checkpoints])

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">4. Results</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {checkpoints.checkpoints.length === 0 ? (
          <div className="text-sm text-muted-foreground py-4 text-center">
            No checkpoints yet. Run a training job to see results here.
          </div>
        ) : (
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground">
              {checkpoints.checkpoints.length} checkpoint(s) saved
            </div>
            <div className="space-y-2">
              {checkpoints.checkpoints.slice(0, 10).map(cp => (
                <div key={cp.name} className="flex items-center justify-between rounded-md border border-border/40 bg-muted/20 px-3 py-2">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <div className="text-xs font-medium truncate">{cp.name}</div>
                      {cp.source === 'turbo' && <Badge variant="warning" size="sm">Turbo</Badge>}
                      {cp.name === bestName && <Badge variant="primary" size="sm">Best</Badge>}
                    </div>
                    <div className="text-[10px] text-muted-foreground">
                      {cp.loss != null && <span>Loss: {cp.loss.toFixed(4)}</span>}
                      {cp.tags && cp.tags.length > 0 && <span className="ml-2">Tags: {cp.tags.join(', ')}</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    <Button size="sm" variant="ghost" onClick={() => checkpoints.handleLoadCheckpoint(cp.name, addToast)}>
                      Load
                    </Button>
                    <Button size="sm" variant="ghost" className="text-destructive" onClick={() => checkpoints.handleDeleteCheckpoint(cp.name, addToast)}>
                      Delete
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div>
          <div className="text-xs text-muted-foreground mb-2">Recent training jobs</div>
          {checkpoints.jobs.length === 0 ? (
            <div className="text-xs text-muted-foreground/70 py-3 text-center">
              No jobs yet. Start a training job to see activity here.
            </div>
          ) : (
            <div className="space-y-2">
              {checkpoints.jobs.slice(0, 8).map(job => {
                const style = jobStatusStyle(job.status)
                const line = formatJobLine(job)
                return (
                  <div key={job.id} className="flex items-center justify-between rounded-md border border-border/40 px-3 py-2">
                    <div className="min-w-0">
                      <div className="text-xs font-medium truncate">{job.name}</div>
                      {line && <div className="text-[10px] text-muted-foreground truncate">{line}</div>}
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {job.progress > 0 && job.progress < 100 && (
                        <span className="text-[10px] text-muted-foreground tabular-nums">{Math.round(job.progress)}%</span>
                      )}
                      <Badge size="sm" variant={style.variant}>{style.label}</Badge>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 pt-2">
          {checkpoints.checkpoints.length > 0 && (
            <Button size="sm" variant="outline" onClick={onTest}>Test model</Button>
          )}
          <Button size="sm" variant="ghost" onClick={goToTrain}>Train more</Button>
        </div>
      </CardContent>
    </Card>
  )
}
