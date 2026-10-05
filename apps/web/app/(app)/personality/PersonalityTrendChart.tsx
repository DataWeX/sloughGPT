'use client'

import { memo, useMemo } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { cn } from '@sloughgpt/strui'

interface HistoryPoint {
  timestamp: number
  voice?: Record<string, number>
  traits?: Record<string, number>
}

interface PersonalityTrendChartProps {
  /** Chart title — rendered as a Label-role eyebrow (uppercase, tracked). */
  title: string
  history: HistoryPoint[]
  /** Which snapshot map to plot: 'voice' or 'traits'. */
  field: 'voice' | 'traits'
  /** Series keys in draw order (stabilizes colors across renders). */
  keys: string[]
  /** Human labels for keys (VOICE_LABELS / TRAIT_LABELS). */
  labels: Record<string, string>
  className?: string
}

/**
 * Chart Colors tokens (DESIGN_SYSTEM.md ▸ Chart Colors). The tokens are hue
 * triplets, so they resolve through rgb() — matching the LossChart pattern.
 * Keyed by metric so each series keeps one semantic color everywhere.
 */
const SERIES_COLORS: Record<string, string> = {
  warmth: 'rgb(var(--chart-5))',
  confidence: 'rgb(var(--chart-4))',
  humor: 'rgb(var(--chart-3))',
  empathy: 'rgb(var(--chart-2))',
  openness: 'rgb(var(--chart-1))',
  agreeableness: 'rgb(var(--chart-2))',
  conscientiousness: 'rgb(var(--chart-4))',
}

const FALLBACK_COLOR = 'hsl(var(--muted-foreground))'

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`
}

function formatWhen(timestamp: number, sameDay: boolean): string {
  const d = new Date(timestamp * 1000)
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  if (sameDay) return time
  return `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${time}`
}

/**
 * One line chart for the Personality Evolution card. Renders twice (voice,
 * traits) with config only — no per-chart copy-paste.
 *
 * Presentation rules (DESIGN_SYSTEM.md): chart-token series colors (no hex),
 * Label-role title eyebrow, inline legend with the current value readout,
 * sparse horizontal-only grid, caption for interaction context.
 */
export const PersonalityTrendChart = memo(function PersonalityTrendChart({
  title,
  history,
  field,
  keys,
  labels,
  className,
}: PersonalityTrendChartProps) {
  const { rows, latest, sameDay } = useMemo(() => {
    const sorted = [...history].sort(
      (a, b) => Number(a.timestamp) - Number(b.timestamp),
    )
    const dayOf = (p: HistoryPoint | undefined) =>
      p ? new Date(Number(p.timestamp) * 1000).toDateString() : ''
    const isSameDay =
      sorted.length === 0 || dayOf(sorted[0]) === dayOf(sorted[sorted.length - 1])
    const plotted = sorted.map((p) => {
      const values = (p[field] ?? {}) as Record<string, number>
      return { time: formatWhen(Number(p.timestamp), isSameDay), ...values }
    })
    const last = sorted[sorted.length - 1]
    const latest = ((last?.[field] ?? {}) as Record<string, number>) ?? {}
    return { rows: plotted, latest, sameDay: isSameDay }
  }, [history, field])

  return (
    <div className={cn('min-w-0', className)}>
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {title}
        </span>
        <span className="text-xs text-muted-foreground/60">
          Across {history.length} interactions
        </span>
      </div>

      {/* Inline legend with current value — replaces Recharts' bottom legend. */}
      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
        {keys.map((key) => (
          <span key={key} className="flex items-center gap-1.5">
            <span
              className="h-1.5 w-1.5 shrink-0 rounded-full"
              style={{ backgroundColor: SERIES_COLORS[key] ?? FALLBACK_COLOR }}
            />
            <span>{labels[key] ?? key}</span>
            {typeof latest[key] === 'number' && (
              <span className="tabular-nums text-muted-foreground/60">
                {formatPercent(latest[key])}
              </span>
            )}
          </span>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid
            vertical={false}
            strokeDasharray="3 3"
            className="stroke-border/30"
          />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10 }}
            tickLine={false}
            stroke="hsl(var(--muted-foreground))"
            interval="preserveStartEnd"
            minTickGap={32}
          />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={(value: number) => formatPercent(value)}
            tick={{ fontSize: 10 }}
            tickLine={false}
            stroke="hsl(var(--muted-foreground))"
            width={36}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
              fontSize: '12px',
            }}
            labelStyle={{ color: 'hsl(var(--muted-foreground))' }}
            formatter={(value) => formatPercent(Number(value))}
          />
          {keys.map((key) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={SERIES_COLORS[key] ?? FALLBACK_COLOR}
              strokeWidth={1.5}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
})
