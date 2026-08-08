'use client'

import { useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceDot } from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

interface CheckpointLossChartProps {
  checkpoints: Checkpoint[]
}

function truncateName(name: string, max = 16): string {
  if (name.length <= max) return name
  return name.slice(0, max - 3) + '...'
}

export function CheckpointLossChart({ checkpoints }: CheckpointLossChartProps) {
  const chartData = useMemo(() => {
    const withLoss = checkpoints
      .filter(c => c.loss != null && c.loss > 0)
      .map(c => ({
        name: truncateName(c.name),
        fullName: c.name,
        loss: c.loss!,
        epochs: c.epochs_trained ?? 0,
      }))

    withLoss.reverse()
    return withLoss
  }, [checkpoints])

  const bestPoint = useMemo(() => {
    if (chartData.length === 0) return null
    return chartData.reduce((best, d) => (d.loss < best.loss ? d : best))
  }, [chartData])

  if (chartData.length < 2) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Loss trend</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border) / 0.3)" />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 10, fill: 'rgb(var(--muted-foreground))' }}
                interval={Math.max(0, Math.floor(chartData.length / 6))}
                angle={-30}
                textAnchor="end"
                height={40}
              />
              <YAxis
                tick={{ fontSize: 10, fill: 'rgb(var(--muted-foreground))' }}
                width={50}
                tickFormatter={(v: number) => v.toFixed(2)}
              />
              <Tooltip
                contentStyle={{
                  fontSize: 11,
                  background: 'rgb(var(--card))',
                  border: '1px solid rgb(var(--border))',
                  borderRadius: 6,
                }}
                formatter={(value: number) => [
                  Number(value).toFixed(4),
                  'Loss',
                ]}
                labelFormatter={(label: string) => {
                  const item = chartData.find(d => d.name === label)
                  return item?.fullName ?? label
                }}
              />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="rgb(var(--primary))"
                strokeWidth={2}
                dot={{ r: 3, fill: 'rgb(var(--primary))' }}
                activeDot={{ r: 5 }}
              />
              {bestPoint && (
                <ReferenceDot
                  x={bestPoint.name}
                  y={bestPoint.loss}
                  r={6}
                  fill="rgb(var(--success))"
                  stroke="rgb(var(--success))"
                  strokeWidth={2}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[10px] text-muted-foreground/50 mt-2">
          {chartData.length} checkpoints · green dot = best ({bestPoint?.loss.toFixed(4)})
        </p>
      </CardContent>
    </Card>
  )
}
