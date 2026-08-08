'use client'

import { useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

interface TrainingHealthCardProps {
  checkpoints: Checkpoint[]
}

type HealthStatus = 'improving' | 'stagnant' | 'diverging' | 'no-data'

interface HealthResult {
  status: HealthStatus
  message: string
  bestLoss: number | null
  recentTrend: number | null
}

function analyze(checkpoints: Checkpoint[]): HealthResult {
  const withLoss = checkpoints
    .filter(c => c.loss != null && c.loss > 0)
    .map(c => ({ name: c.name, loss: c.loss! }))

  if (withLoss.length < 2) {
    return { status: 'no-data', message: 'Need at least 2 checkpoints with loss to analyze', bestLoss: withLoss[0]?.loss ?? null, recentTrend: null }
  }

  const bestLoss = Math.min(...withLoss.map(c => c.loss))
  const recentCount = Math.min(5, withLoss.length)
  const recent = withLoss.slice(0, recentCount)

  const firstHalf = recent.slice(0, Math.floor(recent.length / 2))
  const secondHalf = recent.slice(Math.floor(recent.length / 2))

  const avgFirst = firstHalf.reduce((s, c) => s + c.loss, 0) / firstHalf.length
  const avgSecond = secondHalf.reduce((s, c) => s + c.loss, 0) / secondHalf.length
  const trend = avgSecond - avgFirst

  if (trend < -0.01) {
    return {
      status: 'improving',
      message: `Loss trending down (${trend.toFixed(4)} over last ${recentCount} runs)`,
      bestLoss,
      recentTrend: trend,
    }
  }

  if (trend > 0.05) {
    return {
      status: 'diverging',
      message: `Loss trending up (+${trend.toFixed(4)} over last ${recentCount} runs) — consider lower learning rate`,
      bestLoss,
      recentTrend: trend,
    }
  }

  return {
    status: 'stagnant',
    message: `Loss flat (Δ${trend.toFixed(4)} over last ${recentCount} runs) — try more data or different architecture`,
    bestLoss,
    recentTrend: trend,
  }
}

const STATUS_STYLES: Record<HealthStatus, { badge: string; border: string }> = {
  improving: { badge: 'bg-success/15 text-success', border: 'border-success/30' },
  stagnant: { badge: 'bg-warning/15 text-warning', border: 'border-warning/30' },
  diverging: { badge: 'bg-destructive/15 text-destructive', border: 'border-destructive/30' },
  'no-data': { badge: 'bg-muted text-muted-foreground', border: 'border-border' },
}

const STATUS_LABELS: Record<HealthStatus, string> = {
  improving: 'Improving',
  stagnant: 'Stagnant',
  diverging: 'Diverging',
  'no-data': 'No data',
}

export function TrainingHealthCard({ checkpoints }: TrainingHealthCardProps) {
  const result = useMemo(() => analyze(checkpoints), [checkpoints])
  const styles = STATUS_STYLES[result.status]

  if (checkpoints.length === 0) return null

  return (
    <Card className={styles.border}>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Training health</CardTitle>
        <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${styles.badge}`}>
          {STATUS_LABELS[result.status]}
        </span>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground">{result.message}</p>
        {result.bestLoss != null && (
          <p className="text-[11px] text-muted-foreground/60 mt-1">
            Best loss: {result.bestLoss.toFixed(4)}
          </p>
        )}
      </CardContent>
    </Card>
  )
}
