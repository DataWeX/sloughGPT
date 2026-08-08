'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'

interface TrainingProgressProps {
  job: {
    id: string
    name?: string
    status: string
    progress?: number
    epoch?: number
    totalEpochs?: number
    currentStep?: number
    totalSteps?: number
    loss?: number
    learningRate?: number
    startedAt?: string
    elapsedMs?: number
  } | null
}

function formatElapsed(ms: number): string {
  if (ms <= 0) return '0s'
  const s = Math.floor(ms / 1000)
  const m = Math.floor(s / 60)
  const h = Math.floor(m / 60)
  if (h > 0) return `${h}h ${m % 60}m`
  if (m > 0) return `${m}m ${s % 60}s`
  return `${s}s`
}

function formatETA(ms: number): string {
  if (ms <= 0 || !isFinite(ms)) return '—'
  return formatElapsed(ms)
}

function progressColor(pct: number): string {
  if (pct < 30) return 'bg-primary'
  if (pct < 70) return 'bg-primary'
  return 'bg-success'
}

export function TrainingProgress({ job }: TrainingProgressProps) {
  if (!job || (job.status !== 'running' && job.status !== 'queued')) return null

  const pct = job.progress ?? 0
  const elapsed = job.elapsedMs ?? 0
  const eta = pct > 0 && pct < 100 ? (elapsed / pct) * (100 - pct) : 0

  return (
    <Card data-testid="training-progress">
      <CardHeader className="flex flex-row items-center justify-between py-3">
        <div className="flex items-center gap-2">
          <CardTitle className="text-base">Training Progress</CardTitle>
          <Badge
            variant={job.status === 'running' ? 'default' : 'secondary'}
            className="text-[10px] px-1.5 py-0 h-4"
          >
            {job.status}
          </Badge>
        </div>
        {job.name && (
          <span className="text-[11px] text-muted-foreground truncate max-w-[200px]">{job.name}</span>
        )}
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1.5">
          <div className="flex justify-between text-[11px] text-muted-foreground">
            <span>
              {job.epoch != null && job.totalEpochs != null
                ? `Epoch ${job.epoch}/${job.totalEpochs}`
                : job.currentStep != null && job.totalSteps != null
                  ? `Step ${job.currentStep.toLocaleString()}/${job.totalSteps.toLocaleString()}`
                  : 'Starting...'}
            </span>
            <span>{Math.round(pct)}%</span>
          </div>
          <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={`h-full rounded-full transition-all duration-500 ${progressColor(pct)}`}
              style={{ width: `${Math.min(pct, 100)}%` }}
            />
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2 text-[11px]">
          <div className="space-y-0.5">
            <span className="text-muted-foreground/50">Loss</span>
            <p className="font-mono text-foreground">
              {job.loss != null ? job.loss.toFixed(4) : '—'}
            </p>
          </div>
          <div className="space-y-0.5">
            <span className="text-muted-foreground/50">Elapsed</span>
            <p className="font-mono text-foreground">{formatElapsed(elapsed)}</p>
          </div>
          <div className="space-y-0.5">
            <span className="text-muted-foreground/50">ETA</span>
            <p className="font-mono text-foreground">{formatETA(eta)}</p>
          </div>
        </div>

        {job.learningRate != null && (
          <div className="text-[10px] text-muted-foreground/50">
            LR: {job.learningRate.toExponential(2)}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
