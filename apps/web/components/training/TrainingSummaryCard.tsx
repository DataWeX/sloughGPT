'use client'

import { useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'
import { formatDuration } from './formatDuration'

interface TrainingSummaryCardProps {
  checkpoints: Checkpoint[]
}

interface Stat {
  label: string
  value: string
}

function computeStats(checkpoints: Checkpoint[]): Stat[] {
  if (checkpoints.length === 0) return []

  const withLoss = checkpoints.filter(c => c.loss != null && c.loss > 0)
  const withDuration = checkpoints.filter(c => c.training_duration_s != null && c.training_duration_s > 0)
  const withVocab = checkpoints.filter(c => c.vocab_size != null && c.vocab_size > 0)
  const withQuality = checkpoints.filter(c => c.avg_quality != null && c.avg_quality > 0)

  const bestLoss = withLoss.length > 0 ? Math.min(...withLoss.map(c => c.loss!)) : null
  const avgLoss = withLoss.length > 0 ? withLoss.reduce((s, c) => s + c.loss!, 0) / withLoss.length : null
  const totalDuration = withDuration.reduce((s, c) => s + c.training_duration_s!, 0)
  const bestDuration = withDuration.length > 0 ? Math.min(...withDuration.map(c => c.training_duration_s!)) : null
  const avgQuality = withQuality.length > 0 ? withQuality.reduce((s, c) => s + c.avg_quality!, 0) / withQuality.length : null

  const modelTypes = new Map<string, number>()
  for (const c of checkpoints) {
    const t = c.model_type || c.lineage || 'unknown'
    modelTypes.set(t, (modelTypes.get(t) || 0) + 1)
  }
  const topModel = [...modelTypes.entries()].sort((a, b) => b[1] - a[1])[0]

  const stats: Stat[] = [
    { label: 'Total checkpoints', value: String(checkpoints.length) },
  ]

  if (bestLoss != null) stats.push({ label: 'Best loss', value: bestLoss.toFixed(4) })
  if (avgLoss != null) stats.push({ label: 'Avg loss', value: avgLoss.toFixed(4) })
  if (withLoss.length > 1) {
    const spread = Math.max(...withLoss.map(c => c.loss!)) - bestLoss!
    stats.push({ label: 'Loss spread', value: spread.toFixed(4) })
  }
  if (totalDuration > 0) stats.push({ label: 'Total training time', value: formatDuration(totalDuration) })
  if (bestDuration != null) stats.push({ label: 'Fastest run', value: formatDuration(bestDuration) })
  if (avgQuality != null) stats.push({ label: 'Avg quality', value: `${avgQuality.toFixed(1)}/5` })
  if (withVocab.length > 0) {
    const maxVocab = Math.max(...withVocab.map(c => c.vocab_size!))
    stats.push({ label: 'Max vocab size', value: String(maxVocab) })
  }
  if (topModel && modelTypes.size > 1) {
    stats.push({ label: 'Top model type', value: `${topModel[0]} (${topModel[1]})` })
  }

  return stats
}

export function TrainingSummaryCard({ checkpoints }: TrainingSummaryCardProps) {
  const stats = useMemo(() => computeStats(checkpoints), [checkpoints])

  if (stats.length === 0) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Training summary</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {stats.map(s => (
            <div key={s.label} className="space-y-0.5">
              <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">{s.label}</p>
              <p className="text-sm font-mono font-medium">{s.value}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
