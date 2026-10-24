'use client'

import { Badge } from '@sloughgpt/strui'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

/** Canonical outcome-tracker run shape (GET /settings/training/runs). */
export interface TrainingRun {
  run_id: string
  timestamp: number
  model?: string
  method?: string
  epochs?: number
  final_loss?: number
  best_loss?: number
  perplexity?: number
  quality_score?: number
  training_time_s?: number
  converged?: boolean
  early_stopped?: boolean
  dataset?: string
  tags?: string[]
  notes?: string
  bookmarked?: boolean
}

/** Short "Jan 5" label for chart axes (timestamps are unix seconds). */
export function formatRunDate(ts: number): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

/** Quality score badge — green ≥0.8, yellow ≥0.6, red below. */
export function QualityBadge({ score }: { score?: number }) {
  if (score === undefined || score === null) return null
  const color =
    score >= 0.8
      ? 'bg-green-100 text-green-800'
      : score >= 0.6
        ? 'bg-yellow-100 text-yellow-800'
        : 'bg-red-100 text-red-800'
  return <Badge className={color}>{Math.round(score * 100)}%</Badge>
}

/** Run lifecycle badge shared by runs list, trends table, analytics recents. */
export function RunStatusBadge({ converged, earlyStopped }: { converged?: boolean; earlyStopped?: boolean }) {
  if (converged) return <Badge className="bg-green-100 text-green-800 text-xs">converged</Badge>
  if (earlyStopped) return <Badge variant="destructive" className="text-[10px]">early stopped</Badge>
  return <Badge variant="secondary" className="text-[10px]">completed</Badge>
}

/** Improving / declining / stable trend icon. */
export function TrendIcon({ trend, className }: { trend: string; className?: string }) {
  if (trend === 'improving') return <TrendingUp className={className ?? 'h-4 w-4 text-green-500'} />
  if (trend === 'declining') return <TrendingDown className={className ?? 'h-4 w-4 text-red-500'} />
  return <Minus className={className ?? 'h-4 w-4 text-muted-foreground'} />
}

interface TooltipEntry {
  name: string
  value: number
  color: string
}

/** Single recharts tooltip used by all training charts. */
export function TrainingChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean
  payload?: TooltipEntry[]
  label?: string
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-card border border-border rounded-lg px-3 py-2 shadow-lg text-xs">
      <p className="font-medium mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </p>
      ))}
    </div>
  )
}
