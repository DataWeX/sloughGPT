'use client'

import { useMemo } from 'react'
import { cn, ActionCard } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

interface TrainingHealthCardProps {
  checkpoints: Checkpoint[]
  loading?: boolean
}

type HealthStatus = 'improving' | 'stagnant' | 'diverging' | 'no-data'

interface HealthResult {
  status: HealthStatus
  message: string
  bestLoss: number | null
  recentTrend: number | null
  avgQuality: number | null
  volatility: number | null
  consecutiveDown: number
  consecutiveUp: number
  recommendation: string
  lossHistory: number[]
}

function analyze(checkpoints: Checkpoint[]): HealthResult {
  const withLoss = checkpoints
    .filter(c => c.loss != null && c.loss > 0)
    .map(c => ({ name: c.name, loss: c.loss! }))

  const withQuality = checkpoints.filter(c => c.avg_quality != null && c.avg_quality > 0)
  const avgQuality = withQuality.length > 0
    ? withQuality.reduce((s, c) => s + c.avg_quality!, 0) / withQuality.length
    : null

  const emptyResult: HealthResult = {
    status: 'no-data',
    message: 'Need at least 2 checkpoints with loss to analyze',
    bestLoss: withLoss[0]?.loss ?? null,
    recentTrend: null,
    avgQuality,
    volatility: null,
    consecutiveDown: 0,
    consecutiveUp: 0,
    recommendation: 'Train more checkpoints to enable health analysis',
    lossHistory: [],
  }

  if (withLoss.length < 2) return emptyResult

  const bestLoss = Math.min(...withLoss.map(c => c.loss))
  const lossHistory = withLoss.map(c => c.loss)

  // Volatility: std dev of loss differences
  const diffs: number[] = []
  for (let i = 1; i < lossHistory.length; i++) {
    diffs.push(lossHistory[i] - lossHistory[i - 1])
  }
  const meanDiff = diffs.reduce((s, d) => s + d, 0) / diffs.length
  const volatility = Math.sqrt(diffs.reduce((s, d) => s + (d - meanDiff) ** 2, 0) / diffs.length)

  // Consecutive runs
  let consecutiveDown = 0
  let consecutiveUp = 0
  for (let i = diffs.length - 1; i >= 0; i--) {
    if (diffs[i] < -0.001) consecutiveDown++
    else if (diffs[i] > 0.001) consecutiveUp++
    else break
  }

  // Trend analysis (last 5 runs)
  const recentCount = Math.min(5, withLoss.length)
  const recent = withLoss.slice(0, recentCount)

  const firstHalf = recent.slice(0, Math.floor(recent.length / 2))
  const secondHalf = recent.slice(Math.floor(recent.length / 2))

  const avgFirst = firstHalf.reduce((s, c) => s + c.loss, 0) / firstHalf.length
  const avgSecond = secondHalf.reduce((s, c) => s + c.loss, 0) / secondHalf.length
  const trend = avgSecond - avgFirst

  // Generate recommendation
  let recommendation: string
  let status: HealthStatus
  let message: string

  if (trend < -0.01) {
    status = 'improving'
    message = `Loss trending down (${trend.toFixed(4)} over last ${recentCount} runs)`
    if (volatility > 0.1) {
      recommendation = 'Training is improving but volatile. Consider lowering learning rate or increasing batch size.'
    } else if (consecutiveDown >= 3) {
      recommendation = 'Strong consistent improvement. Continue training or try more epochs.'
    } else {
      recommendation = 'Training is progressing well. Keep going or try a larger model.'
    }
  } else if (trend > 0.05) {
    status = 'diverging'
    message = `Loss trending up (+${trend.toFixed(4)} over last ${recentCount} runs)`
    if (consecutiveUp >= 3) {
      recommendation = 'Model is consistently diverging. Lower learning rate (try 1e-4) or reduce batch size.'
    } else if (volatility > 0.15) {
      recommendation = 'High volatility with upward trend. Reduce learning rate and add warmup steps.'
    } else {
      recommendation = 'Consider lower learning rate or different architecture.'
    }
  } else {
    status = 'stagnant'
    message = `Loss flat (Δ${trend.toFixed(4)} over last ${recentCount} runs)`
    if (withLoss.length > 10) {
      recommendation = 'Training has plateaued. Try more data, different architecture, or increase model capacity.'
    } else {
      recommendation = 'Loss may still be converging. Train more checkpoints before deciding.'
    }
  }

  return {
    status,
    message,
    bestLoss,
    recentTrend: trend,
    avgQuality,
    volatility,
    consecutiveDown,
    consecutiveUp,
    recommendation,
    lossHistory,
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

function MiniSparkline({ data, className }: { data: number[]; className?: string }) {
  if (data.length < 2) return null

  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const height = 24
  const width = 80

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width
    const y = height - ((v - min) / range) * height
    return `${x},${y}`
  }).join(' ')

  return (
    <svg width={width} height={height} className={className} aria-hidden="true">
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-muted-foreground/40"
      />
      {data.length > 0 && (
        <circle
          cx={(data.length - 1) / (data.length - 1) * width}
          cy={height - ((data[data.length - 1] - min) / range) * height}
          r="2"
          fill="currentColor"
          className="text-muted-foreground/60"
        />
      )}
    </svg>
  )
}

export function TrainingHealthCard({ checkpoints, loading }: TrainingHealthCardProps) {
  const result = useMemo(() => analyze(checkpoints), [checkpoints])
  const styles = STATUS_STYLES[result.status]

  if (loading && checkpoints.length === 0) {
    return (
      <ActionCard
        title="Training health"
        actions={
          <div className="h-5 w-16 animate-pulse rounded-full bg-muted" />
        }
      >
          <div className="space-y-2">
            <div className="h-4 w-48 animate-pulse rounded bg-muted" />
            <div className="h-3 w-32 animate-pulse rounded bg-muted" />
          </div>
      </ActionCard>
    )
  }

  if (checkpoints.length === 0) return null

  return (
    <ActionCard
      title="Training health"
      actions={
        <span className={cn('text-[10px] px-2 py-0.5 rounded-full font-medium', styles.badge)}>
          {STATUS_LABELS[result.status]}
        </span>
      }
      className={styles.border}
    >
      <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm text-muted-foreground">{result.message}</p>
          <p className="text-xs text-muted-foreground/60 mt-0.5">{result.recommendation}</p>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground/60 mt-1.5">
            {result.bestLoss != null && (
              <span>Best loss: {result.bestLoss.toFixed(4)}</span>
            )}
            {result.volatility != null && (
              <span>Volatility: {result.volatility.toFixed(4)}</span>
            )}
            {result.consecutiveDown > 0 && (
              <span className="text-success">{result.consecutiveDown} down</span>
            )}
            {result.consecutiveUp > 0 && (
              <span className="text-destructive">{result.consecutiveUp} up</span>
            )}
            {result.avgQuality != null && (
              <span>Data quality: {result.avgQuality.toFixed(1)}/5</span>
            )}
          </div>
        </div>
        {result.lossHistory.length >= 2 && (
          <MiniSparkline data={result.lossHistory} className="text-muted-foreground flex-shrink-0" />
        )}
      </div>
    </ActionCard>
  )
}
