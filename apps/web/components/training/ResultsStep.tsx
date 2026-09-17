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

function jobStatusStyle(status: string): {
  label: string
  variant: 'success' | 'warning' | 'error' | 'secondary'
} {
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

function QualitySparkline({ values }: { values: number[] }) {
  if (values.length < 2) return null
  const max = Math.max(...values, 5)
  const min = Math.min(...values, 0)
  const range = max - min || 1
  const h = 32
  const w = 120
  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w
      const y = h - ((v - min) / range) * h
      return `${x},${y}`
    })
    .join(' ')
  const trend = values[values.length - 1] >= values[0]
  return (
    <div className="flex items-center gap-2">
      <svg width={w} height={h} className="shrink-0">
        <polyline
          points={points}
          fill="none"
          stroke={trend ? 'hsl(142,76%,36%)' : 'hsl(0,84%,60%)'}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span className={`text-[10px] font-medium ${trend ? 'text-green-600' : 'text-red-500'}`}>
        {trend ? 'Improving' : 'Declining'}
      </span>
    </div>
  )
}

export function ResultsStep({ checkpoints, goToTrain, onTest, addToast }: ResultsStepProps) {
  const bestName = useMemo(() => {
    const withLoss = checkpoints.checkpoints.filter((c) => c.loss != null && c.loss > 0)
    if (withLoss.length === 0) return null
    return withLoss.reduce((min, c) => (c.loss! < min.loss! ? c : min), withLoss[0]).name
  }, [checkpoints.checkpoints])

  const qualityTrend = useMemo(() => {
    return checkpoints.jobs
      .filter((j) => j.status === 'completed' && j.avg_quality != null)
      .slice(-10)
      .map((j) => j.avg_quality!)
  }, [checkpoints.jobs])

  const completedJobs = useMemo(() => {
    return checkpoints.jobs
      .filter((j) => j.status === 'completed' || j.status === 'failed')
      .slice(0, 5)
  }, [checkpoints.jobs])

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">4. Results</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {qualityTrend.length >= 2 && (
          <div className="rounded-lg border border-border/40 bg-muted/20 px-3 py-2">
            <div className="text-[10px] text-muted-foreground/60 mb-1">Quality trend</div>
            <QualitySparkline values={qualityTrend} />
          </div>
        )}

        {completedJobs.length > 0 && (
          <div>
            <div className="text-[10px] text-muted-foreground/60 mb-1.5">Recent runs</div>
            <div className="space-y-1">
              {completedJobs.map((job) => {
                const style = jobStatusStyle(job.status)
                return (
                  <div
                    key={job.id}
                    className="flex items-center justify-between rounded-lg border border-border/40 bg-muted/20 px-2.5 py-1.5"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="text-[11px] font-medium truncate">{job.name}</div>
                      <div className="text-[9px] text-muted-foreground/60">
                        {job.method && <span>{job.method}</span>}
                        {job.avg_quality != null && (
                          <span className="ml-1.5">Quality: {job.avg_quality.toFixed(1)}/5</span>
                        )}
                        {job.loss != null && (
                          <span className="ml-1.5">Loss: {job.loss.toFixed(4)}</span>
                        )}
                      </div>
                    </div>
                    <Badge size="sm" variant={style.variant}>
                      {style.label}
                    </Badge>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {checkpoints.checkpoints.length === 0 ? (
          <div className="text-[11px] text-muted-foreground/60 py-4 text-center">
            No checkpoints yet. Run a training job to see results here.
          </div>
        ) : (
          <div className="space-y-2">
            <div className="text-[10px] text-muted-foreground/60">
              {checkpoints.checkpoints.length} checkpoint(s) saved
            </div>
            <div className="space-y-1.5">
              {checkpoints.checkpoints.slice(0, 5).map((cp) => (
                <div
                  key={cp.name}
                  className="flex items-center justify-between rounded-lg border border-border/40 bg-muted/20 px-2.5 py-2"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <div className="text-[11px] font-medium truncate">{cp.name}</div>
                      {cp.source === 'turbo' && (
                        <Badge variant="warning" size="sm">
                          Turbo
                        </Badge>
                      )}
                      {cp.name === bestName && (
                        <Badge
                          variant="outline"
                          size="sm"
                          className="text-primary border-primary/30"
                        >
                          Best
                        </Badge>
                      )}
                    </div>
                    <div className="text-[9px] text-muted-foreground/60 tabular-nums">
                      {cp.loss != null && <span>Loss: {cp.loss.toFixed(4)}</span>}
                      {cp.avg_quality != null && (
                        <span className="ml-2">Quality: {cp.avg_quality.toFixed(1)}/5</span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 text-[10px]"
                      onClick={() => checkpoints.handleLoadCheckpoint(cp.name, addToast)}
                    >
                      Load
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 text-[10px] text-destructive"
                      onClick={() => checkpoints.handleDeleteCheckpoint(cp.name, addToast)}
                    >
                      Delete
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="flex items-center gap-1.5 pt-1">
          {checkpoints.checkpoints.length > 0 && (
            <Button size="sm" variant="outline" className="h-7 text-[11px]" onClick={onTest}>
              Test model
            </Button>
          )}
          <Button size="sm" variant="ghost" className="h-7 text-[11px]" onClick={goToTrain}>
            Train more
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
