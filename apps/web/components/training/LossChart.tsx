'use client'

import { useMemo, memo } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, ComposedChart } from 'recharts'
import { downloadBlob } from '@/lib/download-utils'
import { IconDownload } from '@sloughgpt/strui'
import { todayDateString } from '@/lib/format-bytes'

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
  onExportData?: () => void
}

function exportLossCSV(data: LossPoint[], rewardData?: RewardPoint[]) {
  const allSteps = new Set<number>()
  data.forEach(d => allSteps.add(d.step))
  rewardData?.forEach(d => allSteps.add(d.step))
  const steps = [...allSteps].sort((a, b) => a - b)

  const headers = ['step', 'train_loss', 'eval_loss', 'reward']
  const rows = steps.map(s => {
    const train = data.find(d => d.step === s && d.type === 'train')
    const eval_ = data.find(d => d.step === s && d.type === 'eval')
    const reward = rewardData?.find(d => d.step === s)
    return [s, train?.value ?? '', eval_?.value ?? '', reward?.value ?? ''].join(',')
  })

  const csv = [headers.join(','), ...rows].join('\n')
  downloadBlob(csv, `loss-data-${todayDateString()}.csv`, 'text/csv')
}

export const LossChart = memo(function LossChart({ data, rewardData, height = 200, showLegend = true, live = false, windowSize = 40, onExportData }: LossChartProps) {
  // Memoize step union computation
  const { steps, hasTrain, hasEval } = useMemo(() => {
    const allSteps = new Set<number>()
    data.forEach(d => allSteps.add(d.step))
    rewardData?.forEach(d => allSteps.add(d.step))
    return {
      steps: [...allSteps].sort((a, b) => a - b),
      hasTrain: data.some(d => d.type === 'train'),
      hasEval: data.some(d => d.type === 'eval'),
    }
  }, [data, rewardData])

  // Sliding window: show last N steps during live training
  const visibleSteps = useMemo(() => {
    if (!live || steps.length <= windowSize) return steps
    return steps.slice(-windowSize)
  }, [live, steps, windowSize])

  // Memoize chart data build — O(visibleSteps * data.length) but stable refs
  const chartData = useMemo(() => {
    // Build lookup maps for O(1) access instead of O(n) find
    const trainMap = new Map<number, number>()
    const evalMap = new Map<number, number>()
    const rewardMap = new Map<number, number>()
    for (const d of data) {
      if (d.type === 'train') trainMap.set(d.step, d.value)
      else evalMap.set(d.step, d.value)
    }
    rewardData?.forEach(d => rewardMap.set(d.step, d.value))

    return visibleSteps.map(s => {
      const point: Record<string, number | undefined> = { step: s }
      const train = trainMap.get(s)
      const eval_ = evalMap.get(s)
      const reward = rewardMap.get(s)
      if (train !== undefined) point['train'] = train
      if (eval_ !== undefined) point['eval'] = eval_
      if (reward !== undefined) point['reward'] = reward
      return point
    })
  }, [visibleSteps, data, rewardData])

  const hasReward = Boolean(rewardData?.length)

  if (!data.length && !rewardData?.length) return null

  const Chart = hasReward ? ComposedChart : LineChart

  return (
    <div className="w-full relative group">
      {(data.length > 5) && (
        <button
          onClick={() => onExportData ?? exportLossCSV(data, rewardData)}
          className="absolute top-1 right-1 z-10 h-5 w-5 flex items-center justify-center rounded opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity text-muted-foreground hover:text-foreground hover:bg-muted/60"
          aria-label="Export chart data as CSV"
        >
          <IconDownload className="h-3 w-3" />
        </button>
      )}
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
            labelFormatter={(label) => `Step ${label}`}
            formatter={(value: number, name: string) => [value.toFixed(4), name]}
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
})
