'use client'

import { useCallback, useMemo, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

interface BestCheckpointCardProps {
  checkpoints: Checkpoint[]
  onLoad?: (name: string) => void
}

function fmtDuration(s?: number | null): string {
  if (s == null) return '—'
  if (s < 60) return `${s.toFixed(0)}s`
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}

function scoreCheckpoint(c: Checkpoint): number | null {
  const valLoss = c.final_val_loss
  const trainLoss = c.loss
  const loss = valLoss ?? trainLoss
  if (loss == null || loss <= 0) return null

  let score = loss

  if (c.verdict === 'Good') score *= 0.85
  else if (c.verdict === 'Excellent') score *= 0.7

  if (c.training_duration_s != null && c.training_duration_s > 0 && c.epochs_trained != null && c.epochs_trained > 0) {
    const stepsPerEpoch = c.steps ?? 0
    if (stepsPerEpoch > 0) {
      const efficiency = stepsPerEpoch / c.training_duration_s
      if (efficiency > 1) score *= 0.95
    }
  }

  return score
}

function findBest(checkpoints: Checkpoint[]): Checkpoint | null {
  const scored = checkpoints
    .map(c => ({ cp: c, score: scoreCheckpoint(c) }))
    .filter((x): x is { cp: Checkpoint; score: number } => x.score != null)
  if (scored.length === 0) return null
  return scored.reduce((best, x) => (x.score < best.score ? x : best)).cp
}

export function BestCheckpointCard({ checkpoints, onLoad }: BestCheckpointCardProps) {
  const [loading, setLoading] = useState(false)
  const best = useMemo(() => findBest(checkpoints), [checkpoints])

  const handleLoad = useCallback(async () => {
    if (!best || !onLoad) return
    setLoading(true)
    try {
      onLoad(best.name)
    } finally {
      setLoading(false)
    }
  }, [best, onLoad])

  if (!best) return null

  const bestScore = scoreCheckpoint(best)
  const lossDiff = checkpoints
    .filter(c => c.name !== best.name)
    .reduce((min, c) => {
      const s = scoreCheckpoint(c)
      return s != null && bestScore != null ? Math.min(min, s - bestScore) : min
    }, Infinity)

  return (
    <Card className="border-success/30 bg-success/[0.03]">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-full bg-success" />
          Best checkpoint
        </CardTitle>
        {onLoad && (
          <Button
            size="sm"
            onClick={handleLoad}
            disabled={loading || best.is_loaded}
            className="h-7 text-[11px]"
          >
            {loading ? 'Loading...' : best.is_loaded ? 'Loaded' : 'Use this model'}
          </Button>
        )}
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <p className="text-sm font-medium truncate">{best.name}</p>
          <div className="flex flex-wrap gap-2">
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-success/10 text-success font-medium">
              {best.final_val_loss != null ? `val_loss ${best.final_val_loss.toFixed(4)}` : `loss ${best.loss?.toFixed(4)}`}
            </span>
            {best.verdict && (
              <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                best.verdict === 'Excellent' ? 'bg-success/10 text-success' :
                best.verdict === 'Good' ? 'bg-primary/10 text-primary' :
                'bg-muted text-muted-foreground'
              }`}>
                {best.verdict}
              </span>
            )}
            {best.epochs_trained != null && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                {best.epochs_trained} epochs
              </span>
            )}
            {best.training_duration_s != null && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                {fmtDuration(best.training_duration_s)}
              </span>
            )}
            {best.vocab_size != null && best.vocab_size > 0 && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                vocab {best.vocab_size}
              </span>
            )}
            {best.training_dataset && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                {best.training_dataset}
              </span>
            )}
          </div>
          {lossDiff < Infinity && lossDiff > 0 && (
            <p className="text-[11px] text-muted-foreground/70">
              {lossDiff.toFixed(4)} lower loss than next best
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
