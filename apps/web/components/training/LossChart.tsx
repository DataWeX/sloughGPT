'use client'

import { useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, ComposedChart } from 'recharts'

export interface LossPoint {
  step: number
  value: number
  type: 'train' | 'eval'
}

export interface RewardPoint {
  step: number
  value: number
}

export interface LossChartProps {
  data: LossPoint[]
  rewardData?: RewardPoint[]
  height?: number
  showLegend?: boolean
  live?: boolean
  windowSize?: number
}

export function LossChart({ data, rewardData, height = 200, showLegend = true, live = false, windowSize = 40 }: LossChartProps) {
  const hasReward = Boolean(rewardData?.length)

  // Build union of all steps
  const allSteps = new Set<number>()
  data.forEach(d => allSteps.add(d.step))
  rewardData?.forEach(d => allSteps.add(d.step))
  const steps = [...allSteps].sort((a, b) => a - b)

  // Sliding window: show last N steps during live training
  const visibleSteps = useMemo(() => {
    if (!live || steps.length <= windowSize) return steps
    return steps.slice(-windowSize)
  }, [live, steps, windowSize])

  if (!data.length && !rewardData?.length) return null

  const chartData = visibleSteps.map(s => {
    const point: Record<string, number | undefined> = { step: s }
    const train = data.find(d => d.step === s && d.type === 'train')
    const eval_ = data.find(d => d.step === s && d.type === 'eval')
    const reward = rewardData?.find(d => d.step === s)
    if (train) point['train'] = train.value
    if (eval_) point['eval'] = eval_.value
    if (reward) point['reward'] = reward.value
    return point
  })

  const hasTrain = data.some(d => d.type === 'train')
  const hasEval = data.some(d => d.type === 'eval')

  const Chart = hasReward ? ComposedChart : LineChart

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <Chart data={chartData} margin={{ top: 8, right: hasReward ? 48 : 8, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-border/30" />
          <XAxis
            dataKey="step"
            tick={{ fontSize: 11 }}
            stroke="hsl(var(--muted-foreground))"
            tickLine={false}
            label={{ value: 'Step', position: 'insideBottomRight', offset: -4, style: { fontSize: 10, fill: 'hsl(var(--muted-foreground))' } }}
          />
          <YAxis
            yAxisId="loss"
            tick={{ fontSize: 11 }}
            stroke="hsl(var(--muted-foreground))"
            tickLine={false}
            domain={['auto', 'auto']}
            label={{ value: 'Loss', angle: -90, position: 'insideLeft', offset: 4, style: { fontSize: 10, fill: 'hsl(var(--muted-foreground))' } }}
          />
          {hasReward && (
            <YAxis
              yAxisId="reward"
              orientation="right"
              tick={{ fontSize: 11 }}
              stroke="rgb(var(--success))"
              tickLine={false}
              domain={['auto', 'auto']}
              label={{ value: 'Reward', angle: 90, position: 'insideRight', offset: 4, style: { fontSize: 10, fill: 'rgb(var(--success))' } }}
            />
          )}
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--background))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '4px',
              fontSize: '12px',
            }}
            labelStyle={{ color: 'hsl(var(--foreground))' }}
          />
          {showLegend && <Legend />}
          {hasTrain && (
            <Line
              yAxisId="loss"
              type="monotone"
              dataKey="train"
              stroke="hsl(var(--primary))"
              strokeWidth={1.5}
              dot={false}
              name="Train loss"
              connectNulls
            />
          )}
          {hasEval && (
            <Line
              yAxisId="loss"
              type="monotone"
              dataKey="eval"
              stroke="hsl(var(--warning))"
              strokeWidth={1.5}
              dot={false}
              name="Validation loss"
              connectNulls
            />
          )}
          {hasReward && (
            <Line
              yAxisId="reward"
              type="monotone"
              dataKey="reward"
              stroke="rgb(var(--success))"
              strokeWidth={1.5}
              dot={false}
              name="Reward"
              connectNulls
              strokeDasharray="4 2"
            />
          )}
        </Chart>
      </ResponsiveContainer>
    </div>
  )
}
