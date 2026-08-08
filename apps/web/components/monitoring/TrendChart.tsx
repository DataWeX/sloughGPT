'use client'

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, Area, ComposedChart } from 'recharts'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'

interface TrendPoint {
  ago: string
  health?: number
  mem?: number
  rss?: number
}

interface TrendChartProps {
  liveHealth: LiveHealthSnapshot | null
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

function TrendTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }>; label?: string }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-card border border-border rounded-lg px-3 py-2 shadow-lg text-xs font-mono">
      <p className="text-muted-foreground mb-1">{label} ago</p>
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: entry.color }} />
          <span className="text-muted-foreground">{entry.name}:</span>
          <span className="font-medium">{typeof entry.value === 'number' ? entry.value.toFixed(0) : entry.value}</span>
        </div>
      ))}
    </div>
  )
}

export function TrendChart({ liveHealth }: TrendChartProps) {
  const data = buildPoints(liveHealth)

  if (!data.length) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
        No trend data yet — history builds after a few health snapshots
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
        <XAxis
          dataKey="ago"
          tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }}
          interval="preserveStartEnd"
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          yAxisId="left"
          domain={[0, 100]}
          tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }}
          width={35}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }}
          width={45}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip content={<TrendTooltip />} />
        <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} />
        <Area
          yAxisId="left"
          type="monotone"
          dataKey="health"
          stroke="rgb(var(--success))"
          fill="rgb(var(--success))"
          fillOpacity={0.08}
          strokeWidth={1.5}
          dot={false}
          name="Health"
        />
        <Line
          yAxisId="left"
          type="monotone"
          dataKey="mem"
          stroke="rgb(var(--warning))"
          strokeWidth={1.5}
          dot={false}
          name="Memory %"
          strokeDasharray="4 2"
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="rss"
          stroke="rgb(var(--primary))"
          strokeWidth={1.5}
          dot={false}
          name="Process RSS (MB)"
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
