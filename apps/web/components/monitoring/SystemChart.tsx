'use client'

import { memo } from 'react'
import { Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, Area, ComposedChart } from 'recharts'
import { ChartFrame } from '@/components/charts/ChartFrame'
import { ChartTooltip } from '@/components/charts/ChartTooltip'
import { AXIS, GRID, seriesColor } from '@/components/charts/chart-tokens'
import { usePrefersReducedMotion, chartAnimation } from '@/components/charts/chart-motion'

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

export const SystemChart = memo(function SystemChart({ data, showTokens = true, showLatency = true }: SystemChartProps) {
  const reduced = usePrefersReducedMotion()

  return (
    <ChartFrame
      label="System metrics chart"
      empty={data.length === 0}
      emptyMessage="Waiting for data..."
      srSummary="CPU and memory percent with throughput (tok/s) and latency (ms) over time."
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} opacity={0.5} />
          <XAxis
            dataKey="time"
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
              label={{ value: '%', position: 'insideTopLeft', offset: 10, style: { fontSize: 10, fill: AXIS } }}
            />
            {(showTokens || showLatency) && (
              <YAxis
                yAxisId="right"
                orientation="right"
                tick={{ fontSize: 10, fill: AXIS }}
                width={40}
                tickLine={false}
                axisLine={false}
              />
            )}
          <Tooltip content={<ChartTooltip valueFormatter={(v) => Number(v).toFixed(1)} />} />
          <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} />
          <Area
            {...chartAnimation(reduced)}
            yAxisId="left"
            type="monotone"
            dataKey="cpu"
            stroke={seriesColor(0)}
            fill={seriesColor(0)}
            fillOpacity={0.08}
            strokeWidth={1.5}
            dot={false}
            name="CPU %"
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
          {showTokens && (
            <Line
              {...chartAnimation(reduced)}
              yAxisId="right"
              type="monotone"
              dataKey="tokens"
              stroke={seriesColor(2)}
              strokeWidth={1.5}
              dot={false}
              name="tok/s"
            />
          )}
          {showLatency && (
            <Line
              {...chartAnimation(reduced)}
              yAxisId="right"
              type="monotone"
              dataKey="latency"
              stroke={seriesColor(3)}
              strokeWidth={1}
              dot={false}
              name="Latency ms"
              strokeDasharray="2 2"
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
})
