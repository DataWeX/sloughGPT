'use client'

import { useMemo, memo } from 'react'
import { Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, Area, ComposedChart } from 'recharts'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'
import { ChartFrame } from '@/components/charts/ChartFrame'
import { ChartTooltip } from '@/components/charts/ChartTooltip'
import { AXIS, GRID, seriesColor } from '@/components/charts/chart-tokens'
import { usePrefersReducedMotion, chartAnimation } from '@/components/charts/chart-motion'

interface TrendPoint {
  ago: string
  health?: number
  mem?: number
  rss?: number
}

interface TrendChartProps {
  liveHealth: LiveHealthSnapshot | null
  /** True while the first health snapshot is loading — renders the frame skeleton. */
  loading?: boolean
}

function nearestMemory(ts: number, memory: LiveHealthSnapshot['memory_history']): { system_percent: number; rss_mb: number } | undefined {
  if (!memory.length) return undefined
  let best: { system_percent: number; rss_mb: number } | undefined
  let bestDist = Infinity
  for (const m of memory) {
    const d = Math.abs(m.ts - ts)
    if (d < bestDist) {
      bestDist = d
      best = { system_percent: m.system_percent, rss_mb: m.rss_mb }
    }
  }
  return best
}

function buildPoints(liveHealth: LiveHealthSnapshot | null): TrendPoint[] {
  const health = liveHealth?.health_history ?? []
  const memory = liveHealth?.memory_history ?? []
  const now = Date.now() / 1000
  if (!health.length && !memory.length) return []

  const points: TrendPoint[] = []
  if (health.length) {
    for (const h of health) {
      const m = nearestMemory(h.ts, memory)
      points.push({
        ago: `${Math.max(0, Math.round(now - h.ts))}s`,
        health: h.score,
        mem: m?.system_percent,
        rss: m?.rss_mb,
      })
    }
  } else {
    for (const m of memory) {
      points.push({ ago: `${Math.max(0, Math.round(now - m.ts))}s`, mem: m.system_percent, rss: m.rss_mb })
    }
  }
  return points
}

export const TrendChart = memo(function TrendChart({ liveHealth, loading }: TrendChartProps) {
  const reduced = usePrefersReducedMotion()
  const data = useMemo(() => buildPoints(liveHealth), [liveHealth])

  return (
    <ChartFrame
      label="Health and memory trend chart"
      loading={loading}
      empty={data.length === 0}
      emptyMessage="No trend data yet — history builds after a few health snapshots"
      srSummary={`Health score, memory percent and process RSS across ${data.length} recent snapshots, labelled seconds ago.`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} opacity={0.5} />
          <XAxis
              dataKey="ago"
              tick={{ fontSize: 10, fill: AXIS }}
              interval="preserveStartEnd"
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              yAxisId="left"
              domain={[0, 100]}
              tick={{ fontSize: 10, fill: AXIS }}
              width={35}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              yAxisId="right"
            orientation="right"
              tick={{ fontSize: 10, fill: AXIS }}
              width={45}
              tickLine={false}
              axisLine={false}
            />
          <Tooltip
            content={
              <ChartTooltip
                labelFormatter={(l) => `${l} ago`}
                valueFormatter={(v) => Number(v).toFixed(0)}
              />
            }
          />
          <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} />
          <Area
            {...chartAnimation(reduced)}
            yAxisId="left"
            type="monotone"
            dataKey="health"
            stroke={seriesColor(0)}
            fill={seriesColor(0)}
            fillOpacity={0.08}
            strokeWidth={1.5}
            dot={false}
            name="Health"
          />
          <Line
            {...chartAnimation(reduced)}
            yAxisId="left"
            type="monotone"
            dataKey="mem"
            stroke={seriesColor(1)}
            strokeWidth={1.5}
            dot={false}
            name="Memory %"
            strokeDasharray="4 2"
          />
          <Line
            {...chartAnimation(reduced)}
            yAxisId="right"
            type="monotone"
            dataKey="rss"
            stroke={seriesColor(2)}
            strokeWidth={1.5}
            dot={false}
            name="Process RSS (MB)"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
})
