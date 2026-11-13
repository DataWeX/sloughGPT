'use client'

import { memo, useEffect, useMemo, useRef, useState } from 'react'
import type { MouseEvent as ReactMouseEvent } from 'react'
import { scaleLinear } from 'd3-scale'
import { line as d3Line, curveMonotoneX } from 'd3-shape'
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

/** Fallback plot width until ResizeObserver reports the real container width. */
const DEFAULT_WIDTH = 640

const LOSS_COLOR = 'rgb(var(--primary))'
const EVAL_COLOR = 'rgb(var(--warning))'
const REWARD_COLOR = 'rgb(var(--success))'
const AXIS_COLOR = 'rgb(var(--muted-foreground))'
const GRID_COLOR = 'rgb(var(--border))'

interface ChartPoint {
  step: number
  train?: number
  eval?: number
  reward?: number
}

/** Tick-label precision scales with the visible domain span. */
function formatTick(value: number, span: number): string {
  if (span >= 10) return value.toFixed(0)
  if (span >= 1) return value.toFixed(1)
  const decimals = Math.min(6, Math.max(2, Math.ceil(-Math.log10(span)) + 1))
  return value.toFixed(decimals)
}

/** [min, max] with 8% padding; a degenerate single-value domain gets a fixed pad. */
function paddedDomain(values: number[]): [number, number] {
  if (values.length === 0) return [0, 1]
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  if (lo === hi) {
    const pad = Math.abs(lo) * 0.1 || 0.1
    return [lo - pad, hi + pad]
  }
  const pad = (hi - lo) * 0.08
  return [lo - pad, hi + pad]
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="inline-block h-0.5 w-3 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </span>
  )
}

function TooltipRow({ color, label, value }: { color: string; label: string; value: number }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
      <span className="text-muted-foreground">{label}</span>
      <span className="ml-2 font-mono text-foreground">{value.toFixed(4)}</span>
    </div>
  )
}

export const LossChart = memo(function LossChart({
  data,
  rewardData,
  height = 200,
  showLegend = true,
  live = false,
  windowSize = 40,
  onExportData,
}: LossChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [width, setWidth] = useState(DEFAULT_WIDTH)
  const [hoveredStep, setHoveredStep] = useState<number | null>(null)

  // jsdom stubs ResizeObserver as a no-op, so tests render at DEFAULT_WIDTH.
  useEffect(() => {
    const el = containerRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(entries => {
      const w = entries[0]?.contentRect.width ?? 0
      if (w > 0) setWidth(w)
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

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

  const chartData = useMemo<ChartPoint[]>(() => {
    const trainMap = new Map<number, number>()
    const evalMap = new Map<number, number>()
    const rewardMap = new Map<number, number>()
    for (const d of data) {
      if (d.type === 'train') trainMap.set(d.step, d.value)
      else evalMap.set(d.step, d.value)
    }
    rewardData?.forEach(d => rewardMap.set(d.step, d.value))

    return visibleSteps.map(s => {
      const point: ChartPoint = { step: s }
      const train = trainMap.get(s)
      const eval_ = evalMap.get(s)
      const reward = rewardMap.get(s)
      if (train !== undefined) point.train = train
      if (eval_ !== undefined) point.eval = eval_
      if (reward !== undefined) point.reward = reward
      return point
    })
  }, [visibleSteps, data, rewardData])

  const hasReward = Boolean(rewardData?.length)

  const layout = useMemo(() => {
    const margin = { top: 10, right: hasReward ? 54 : 16, bottom: 26, left: 50 }
    const plotW = Math.max(20, width - margin.left - margin.right)
    const plotH = Math.max(20, height - margin.top - margin.bottom)

    const xDomain: [number, number] =
      visibleSteps.length === 0
        ? [0, 1]
        : visibleSteps.length === 1
          ? [visibleSteps[0] - 1, visibleSteps[0] + 1]
          : [visibleSteps[0], visibleSteps[visibleSteps.length - 1]]

    const lossValues: number[] = []
    const rewardValues: number[] = []
    for (const p of chartData) {
      if (p.train !== undefined) lossValues.push(p.train)
      if (p.eval !== undefined) lossValues.push(p.eval)
      if (p.reward !== undefined) rewardValues.push(p.reward)
    }

    // Linear scales + explicit tick counts are the fix for the old category
    // axis: ticks are evenly spaced multiples of a 1/2/5×10^k step, never
    // "one tick per data point" (the irregular 2,5,6,9,11,14,... axis).
    const xScale = scaleLinear().domain(xDomain).range([margin.left, margin.left + plotW])
    const yScale = scaleLinear()
      .domain(paddedDomain(lossValues))
      .nice(5)
      .range([margin.top + plotH, margin.top])
    const rewardScale = hasReward
      ? scaleLinear()
          .domain(paddedDomain(rewardValues))
          .nice(5)
          .range([margin.top + plotH, margin.top])
      : null

    return {
      margin,
      plotW,
      plotH,
      xScale,
      yScale,
      rewardScale,
      xTicks: xScale.ticks(6),
      yTicks: yScale.ticks(5),
      rewardTicks: rewardScale ? rewardScale.ticks(5) : [],
    }
  }, [visibleSteps, chartData, width, height, hasReward])

  const paths = useMemo(() => {
    const build = (key: 'train' | 'eval' | 'reward') => {
      const points = chartData
        .filter(p => p[key] !== undefined)
        .map(p => ({ step: p.step, v: p[key] as number }))
      if (points.length === 0) return ''
      const scale = key === 'reward' && layout.rewardScale ? layout.rewardScale : layout.yScale
      return (
        d3Line<{ step: number; v: number }>()
          .x(d => layout.xScale(d.step))
          .y(d => scale(d.v))
          .curve(curveMonotoneX)(points) ?? ''
      )
    }
    return { train: build('train'), eval: build('eval'), reward: build('reward') }
  }, [chartData, layout])

  const handleOverlayMove = (event: ReactMouseEvent<SVGRectElement>) => {
    if (visibleSteps.length === 0) return
    const rect = event.currentTarget.getBoundingClientRect()
    const boxWidth = rect.width > 0 ? rect.width : layout.plotW
    const offsetX = (event.clientX - rect.left) * (layout.plotW / boxWidth)
    const hovered = layout.xScale.invert(layout.margin.left + offsetX)
    let nearest = visibleSteps[0]
    for (const s of visibleSteps) {
      if (Math.abs(s - hovered) < Math.abs(nearest - hovered)) nearest = s
    }
    setHoveredStep(nearest)
  }

  const hoveredPoint = hoveredStep !== null ? chartData.find(p => p.step === hoveredStep) : undefined

  if (!data.length && !rewardData?.length) return null

  const { margin, plotW, plotH, xScale, yScale, rewardScale, xTicks, yTicks, rewardTicks } = layout
  const ySpan = yScale.domain()[1] - yScale.domain()[0]
  const rewardSpan = rewardScale ? rewardScale.domain()[1] - rewardScale.domain()[0] : 1
  const axisXEnd = margin.left + plotW
  const axisYEnd = margin.top + plotH
  const hoverX = hoveredPoint ? xScale(hoveredPoint.step) : 0

  return (
    <div ref={containerRef} className="w-full relative group">
      {data.length > 5 && (
        <button
          type="button"
          onClick={() => onExportData ?? exportLossCSV(data, rewardData)}
          className="absolute top-1 right-1 z-10 h-4 w-4 flex items-center justify-center rounded opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity text-muted-foreground/60 hover:text-foreground hover:bg-muted/60"
          aria-label="Export chart data as CSV"
        >
          <IconDownload className="h-2.5 w-2.5" />
        </button>
      )}

      <svg
        data-testid="chart-svg"
        width="100%"
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Training loss over steps"
      >
        {/* grid */}
        <g aria-hidden="true">
          {yTicks.map(t => (
            <line
              key={`gy-${t}`}
              x1={margin.left}
              x2={axisXEnd}
              y1={yScale(t)}
              y2={yScale(t)}
              stroke={GRID_COLOR}
              strokeOpacity={0.3}
              strokeDasharray="3 3"
            />
          ))}
          {xTicks.map(t => (
            <line
              key={`gx-${t}`}
              x1={xScale(t)}
              x2={xScale(t)}
              y1={margin.top}
              y2={axisYEnd}
              stroke={GRID_COLOR}
              strokeOpacity={0.3}
              strokeDasharray="3 3"
            />
          ))}
        </g>

        {/* axes */}
        <line x1={margin.left} x2={axisXEnd} y1={axisYEnd} y2={axisYEnd} stroke={AXIS_COLOR} />
        <line x1={margin.left} x2={margin.left} y1={margin.top} y2={axisYEnd} stroke={AXIS_COLOR} />

        <g data-testid="x-axis">
          {xTicks.map(t => (
            <text
              key={`tx-${t}`}
              x={xScale(t)}
              y={axisYEnd + 14}
              textAnchor="middle"
              fontSize={10}
              fill={AXIS_COLOR}
            >
              {t}
            </text>
          ))}
          <text x={axisXEnd} y={height - 4} textAnchor="end" fontSize={9} fill={AXIS_COLOR}>
            Step
          </text>
        </g>

        <g data-testid="y-axis">
          {yTicks.map(t => (
            <text
              key={`ty-${t}`}
              x={margin.left - 6}
              y={yScale(t) + 3.5}
              textAnchor="end"
              fontSize={10}
              fill={AXIS_COLOR}
            >
              {formatTick(t, ySpan)}
            </text>
          ))}
          <text
            x={14}
            y={margin.top + plotH / 2}
            textAnchor="middle"
            fontSize={9}
            fill={AXIS_COLOR}
            transform={`rotate(-90 14 ${margin.top + plotH / 2})`}
          >
            Loss
          </text>
        </g>

        {rewardScale && (
          <g data-testid="reward-axis">
            <line x1={axisXEnd} x2={axisXEnd} y1={margin.top} y2={axisYEnd} stroke={AXIS_COLOR} />
            {rewardTicks.map(t => (
              <text
                key={`tr-${t}`}
                x={axisXEnd + 6}
                y={rewardScale(t) + 3.5}
                textAnchor="start"
                fontSize={10}
                fill={AXIS_COLOR}
              >
                {formatTick(t, rewardSpan)}
              </text>
            ))}
            <text
              x={width - 14}
              y={margin.top + plotH / 2}
              textAnchor="middle"
              fontSize={9}
              fill={REWARD_COLOR}
              transform={`rotate(90 ${width - 14} ${margin.top + plotH / 2})`}
            >
              Reward
            </text>
          </g>
        )}

        {/* series — connectNulls semantics: gaps are bridged, single path */}
        {hasTrain && paths.train && (
          <path data-testid="train-line" d={paths.train} fill="none" stroke={LOSS_COLOR} strokeWidth={1.5} />
        )}
        {hasEval && paths.eval && (
          <path data-testid="eval-line" d={paths.eval} fill="none" stroke={EVAL_COLOR} strokeWidth={1.5} />
        )}
        {hasReward && paths.reward && (
          <path
            data-testid="reward-line"
            d={paths.reward}
            fill="none"
            stroke={REWARD_COLOR}
            strokeWidth={1.5}
            strokeDasharray="4 2"
          />
        )}

        {/* hover marker */}
        {hoveredPoint && (
          <g aria-hidden="true">
            <line x1={hoverX} x2={hoverX} y1={margin.top} y2={axisYEnd} stroke={GRID_COLOR} strokeDasharray="3 3" />
            {hoveredPoint.train !== undefined && (
              <circle cx={hoverX} cy={yScale(hoveredPoint.train)} r={3} fill={LOSS_COLOR} />
            )}
            {hoveredPoint.eval !== undefined && (
              <circle cx={hoverX} cy={yScale(hoveredPoint.eval)} r={3} fill={EVAL_COLOR} />
            )}
            {hoveredPoint.reward !== undefined && rewardScale && (
              <circle cx={hoverX} cy={rewardScale(hoveredPoint.reward)} r={3} fill={REWARD_COLOR} />
            )}
          </g>
        )}

        {/* hover capture */}
        <rect
          data-testid="hover-overlay"
          x={margin.left}
          y={margin.top}
          width={plotW}
          height={plotH}
          fill="transparent"
          onMouseMove={handleOverlayMove}
          onMouseLeave={() => setHoveredStep(null)}
        />
      </svg>

      {hoveredPoint && (
        <div
          data-testid="chart-tooltip"
          className="pointer-events-none absolute z-10 rounded-md border border-border bg-background px-2 py-1.5 text-[11px] shadow-sm"
          style={{
            left: Math.min(Math.max(xScale(hoveredPoint.step), 64), width - 64),
            top: margin.top + 2,
            transform: 'translateX(-50%)',
          }}
        >
          <div className="mb-1 font-medium text-foreground">Step {hoveredPoint.step}</div>
          {hoveredPoint.train !== undefined && (
            <TooltipRow color={LOSS_COLOR} label="Train loss" value={hoveredPoint.train} />
          )}
          {hoveredPoint.eval !== undefined && (
            <TooltipRow color={EVAL_COLOR} label="Validation loss" value={hoveredPoint.eval} />
          )}
          {hoveredPoint.reward !== undefined && (
            <TooltipRow color={REWARD_COLOR} label="Reward" value={hoveredPoint.reward} />
          )}
        </div>
      )}

      {showLegend && (
        <div
          data-testid="legend"
          className="mt-1 flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-[10px] text-muted-foreground"
        >
          {hasTrain && <LegendSwatch color={LOSS_COLOR} label="Train loss" />}
          {hasEval && <LegendSwatch color={EVAL_COLOR} label="Validation loss" />}
          {hasReward && <LegendSwatch color={REWARD_COLOR} label="Reward" />}
        </div>
      )}
    </div>
  )
})
