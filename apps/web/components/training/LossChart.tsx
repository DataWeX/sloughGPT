'use client'

import { useMemo, useState, memo } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, ComposedChart, Brush } from 'recharts'
import { downloadBlob } from '@/lib/download-utils'
import { IconDownload } from '@sloughgpt/strui'
import { todayDateString } from '@/lib/format-bytes'
import { ChartFrame } from '@/components/charts/ChartFrame'
import { ChartTooltip } from '@/components/charts/ChartTooltip'
import { AXIS, seriesColor } from '@/components/charts/chart-tokens'
import { usePrefersReducedMotion, chartAnimation } from '@/components/charts/chart-motion'

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
  /** Chart-scale loading state — renders the ChartFrame skeleton. */
  loading?: boolean
  /** Chart-scale error state — renders the ChartFrame alert with retry. */
  error?: string | null
  onRetry?: () => void
  /** Progressive-disclosure "detailed view" toggle (range brush). Shown for
   *  series longer than 12 steps; pass false to hide entirely. */
  detailToggle?: boolean
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

export const LossChart = memo(function LossChart({ data, rewardData, height = 200, showLegend = true, live = false, windowSize = 40, onExportData, loading, error, onRetry, detailToggle: detailToggleEnabled = true }: LossChartProps) {
  const reduced = usePrefersReducedMotion()

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
  const empty = !data.length && !rewardData?.length

  // Detailed-view brush (UX_FLOWS.md:352 power-user extra) — progressive
  // disclosure: the toggle only appears on series long enough to navigate.
  const [showBrush, setShowBrush] = useState(false)
  const detailToggleVisible = detailToggleEnabled && steps.length > 12

  const seriesNames = [hasTrain && 'train loss', hasEval && 'validation loss', hasReward && 'reward']
    .filter(Boolean)
    .join(', ')

  const Chart = hasReward ? ComposedChart : LineChart

  return (
    <ChartFrame
      label={hasReward ? 'Loss and reward chart' : 'Loss chart'}
      height={height}
      loading={loading}
      error={error}
      onRetry={onRetry}
      empty={empty}
      emptyMessage="No loss data yet"
      srSummary={empty ? undefined : `Series: ${seriesNames}. ${visibleSteps.length} steps.`}
      action={
        data.length > 5 || detailToggleVisible ? (
          <>
            {detailToggleVisible && (
              <button
                type="button"
                onClick={() => setShowBrush(v => !v)}
                aria-label="Toggle detailed chart view"
                aria-pressed={showBrush}
                className={`text-xs uppercase tracking-wider transition-colors ${showBrush ? 'text-foreground' : 'text-muted-foreground/60 hover:text-foreground'}`}
              >
                Detail
              </button>
            )}
            {data.length > 5 && (
              <button
                type="button"
                onClick={() => (onExportData ? onExportData() : exportLossCSV(data, rewardData))}
                className="h-4 w-4 flex items-center justify-center rounded text-muted-foreground/60 hover:text-foreground hover:bg-muted/60 transition-opacity focus-visible:text-foreground"
                aria-label="Export chart data as CSV"
              >
                <IconDownload className="h-2.5 w-2.5" />
              </button>
            )}
          </>
        ) : undefined
      }
    >
      <ResponsiveContainer width="100%" height={height}>
        <Chart data={chartData} margin={{ top: 8, right: hasReward ? 48 : 8, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-border/30" />
        <XAxis
          dataKey="step"
          tick={{ fontSize: 10 }}
          stroke={AXIS}
          tickLine={false}
          label={{ value: 'Step', position: 'insideBottomRight', offset: -4, style: { fontSize: 9, fill: AXIS } }}
        />
          <YAxis
            yAxisId="loss"
            tick={{ fontSize: 10 }}
            stroke={AXIS}
            tickLine={false}
            domain={['auto', 'auto']}
            label={{ value: 'Loss', angle: -90, position: 'insideLeft', offset: 4, style: { fontSize: 9, fill: AXIS } }}
          />
          {hasReward && (
            <YAxis
              yAxisId="reward"
              orientation="right"
              tick={{ fontSize: 10 }}
              stroke={seriesColor(2)}
              tickLine={false}
              domain={['auto', 'auto']}
              label={{ value: 'Reward', angle: 90, position: 'insideRight', offset: 4, style: { fontSize: 9, fill: seriesColor(2) } }}
            />
          )}
          <Tooltip
            content={<ChartTooltip labelFormatter={(l) => `Step ${l}`} />}
            cursor={{ stroke: 'rgb(var(--border))' }}
          />
          {showLegend && <Legend />}
          {showBrush && (
            <Brush dataKey="step" height={20} stroke={AXIS} fill="rgb(var(--card))" travellerWidth={8} />
          )}
          {hasTrain && (
            <Line
              {...chartAnimation(reduced)}
              yAxisId="loss"
              type="monotone"
              dataKey="train"
              stroke={seriesColor(0)}
              strokeWidth={1.5}
              dot={false}
              name="Train loss"
              connectNulls
            />
          )}
          {hasEval && (
            <Line
              {...chartAnimation(reduced)}
              yAxisId="loss"
              type="monotone"
              dataKey="eval"
              stroke={seriesColor(1)}
              strokeWidth={1.5}
              dot={false}
              name="Validation loss"
              connectNulls
            />
          )}
          {hasReward && (
            <Line
              {...chartAnimation(reduced)}
              yAxisId="reward"
              type="monotone"
              dataKey="reward"
              stroke={seriesColor(2)}
              strokeWidth={1.5}
              dot={false}
              name="Reward"
              connectNulls
              strokeDasharray="4 2"
            />
          )}
        </Chart>
      </ResponsiveContainer>
    </ChartFrame>
  )
})
