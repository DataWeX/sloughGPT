'use client'

import { memo, Suspense } from 'react'
import { ComposedChart, Line, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from '@/lib/recharts-lazy'

interface ChartPoint {
  time: string
  cpu: number
  mem: number
  tokens?: number
  latency?: number
}

interface SystemChartProps {
  data: ChartPoint[]
  showTokens?: boolean
  showLatency?: boolean
}

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }>; label?: string }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-card border border-border rounded-lg px-3 py-2 shadow-lg text-xs font-numeric">
      <p className="text-muted-foreground mb-1">{label}</p>
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: entry.color }} />
          <span className="text-muted-foreground">{entry.name}:</span>
          <span className="font-medium">{typeof entry.value === 'number' ? entry.value.toFixed(1) : entry.value}</span>
        </div>
      ))}
    </div>
  )
}

export const SystemChart = memo(function SystemChart({ data, showTokens = true, showLatency = true }: SystemChartProps) {
  if (!data.length) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
        Waiting for data...
      </div>
    )
  }

  return (
    <Suspense fallback={<div className="h-full flex items-center justify-center text-xs text-muted-foreground">Loading chart...</div>}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
          <XAxis
            dataKey="time"
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
            label={{ value: '%', position: 'insideTopLeft', offset: 10, style: { fontSize: 10, fill: 'var(--muted-foreground)' } }}
          />
        {(showTokens || showLatency) && (
          <YAxis
            yAxisId="right"
            orientation="right"
            tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }}
            width={40}
            tickLine={false}
            axisLine={false}
          />
        )}
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} />
        <Area
          yAxisId="left"
          type="monotone"
          dataKey="cpu"
          stroke="rgb(var(--primary))"
          fill="rgb(var(--primary))"
          fillOpacity={0.08}
          strokeWidth={1.5}
          dot={false}
          name="CPU %"
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
        {showTokens && (
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="tokens"
            stroke="rgb(var(--success))"
            strokeWidth={1.5}
            dot={false}
            name="tok/s"
          />
        )}
        {showLatency && (
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="latency"
            stroke="rgb(var(--destructive))"
            strokeWidth={1}
            dot={false}
            name="Latency ms"
            strokeDasharray="2 2"
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
    </Suspense>
  )
})
