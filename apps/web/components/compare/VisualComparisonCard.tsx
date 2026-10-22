'use client'

import { memo, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from 'recharts'
import { ChartFrame } from '@/components/charts/ChartFrame'
import { ChartTooltip } from '@/components/charts/ChartTooltip'
import { AXIS, GRID, seriesColor } from '@/components/charts/chart-tokens'

interface VisualComparisonCardProps {
  chartData: { name: string; throughput: number; latency: number; memory: number }[]
}

export default memo(function VisualComparisonCard({ chartData }: VisualComparisonCardProps) {
  const maxThroughput = Math.max(...chartData.map(d => d.throughput))
  const maxLatency = Math.max(...chartData.map(d => d.latency))
  const maxMemory = Math.max(...chartData.map(d => d.memory))

  const radarData = useMemo(() => chartData.map(d => ({
    name: d.name,
    throughput: maxThroughput > 0 ? (d.throughput / maxThroughput) * 100 : 0,
    latency: maxLatency > 0 ? ((maxLatency - d.latency) / maxLatency) * 100 : 0,
    memory: maxMemory > 0 ? ((maxMemory - d.memory) / maxMemory) * 100 : 0,
  })), [chartData, maxThroughput, maxLatency, maxMemory])

  if (chartData.length < 2) {
    return (
      <Card>
        <CardHeader className="pb-2"><CardTitle className="text-xs">Visual Comparison</CardTitle></CardHeader>
        <CardContent>
          <ChartFrame label="Model comparison charts" empty emptyMessage="Add at least two models to compare" />
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="pb-2"><CardTitle className="text-xs">Visual Comparison</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <div>
          <p className="text-[10px] text-muted-foreground/60 mb-1.5">Throughput (tok/s) — higher is better</p>
          <ChartFrame label="Throughput by model" height={176}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
                <XAxis type="number" tick={{ fontSize: 9 }} stroke={AXIS} />
                <YAxis type="category" dataKey="name" width={80} tick={{ fontSize: 9 }} stroke={AXIS} />
                <Tooltip content={<ChartTooltip valueFormatter={(v) => `${Number(v).toFixed(1)} tok/s`} />} />
                <Bar dataKey="throughput" name="Throughput" fill={seriesColor(0)} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartFrame>
        </div>
        <div>
          <p className="text-[10px] text-muted-foreground/60 mb-1.5">Average latency (ms) — lower is better</p>
          <ChartFrame label="Latency by model" height={176}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
                <XAxis type="number" tick={{ fontSize: 9 }} stroke={AXIS} />
                <YAxis type="category" dataKey="name" width={80} tick={{ fontSize: 9 }} stroke={AXIS} hide />
                <Tooltip content={<ChartTooltip valueFormatter={(v) => `${Number(v).toFixed(0)} ms`} />} />
                <Bar dataKey="latency" name="Latency" fill={seriesColor(1)} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartFrame>
        </div>
        <div>
          <p className="text-[10px] text-muted-foreground/60 mb-1.5">Memory usage (MB) — lower is better</p>
          <ChartFrame label="Memory usage by model" height={176}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
                <XAxis type="number" tick={{ fontSize: 9 }} stroke={AXIS} />
                <YAxis type="category" dataKey="name" width={80} tick={{ fontSize: 9 }} stroke={AXIS} hide />
                <Tooltip content={<ChartTooltip valueFormatter={(v) => `${Number(v).toFixed(0)} MB`} />} />
                <Bar dataKey="memory" name="Memory" fill={seriesColor(2)} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartFrame>
        </div>
        {chartData.length >= 3 && (
          <div>
            <p className="text-[10px] text-muted-foreground/60 mb-1.5">Overall profile (normalized, higher is better)</p>
            <ChartFrame label="Overall model profile radar" height={224}>
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid stroke={GRID} />
                  <PolarAngleAxis dataKey="name" tick={{ fontSize: 9 }} />
                  <PolarRadiusAxis tick={{ fontSize: 8 }} domain={[0, 100]} />
                  <Radar name="Throughput" dataKey="throughput" stroke={seriesColor(0)} fill={seriesColor(0)} fillOpacity={0.15} />
                  <Radar name="Latency" dataKey="latency" stroke={seriesColor(1)} fill={seriesColor(1)} fillOpacity={0.15} />
                  <Radar name="Memory" dataKey="memory" stroke={seriesColor(2)} fill={seriesColor(2)} fillOpacity={0.15} />
                  <Legend wrapperStyle={{ fontSize: 9 }} />
                </RadarChart>
              </ResponsiveContainer>
            </ChartFrame>
          </div>
        )}
      </CardContent>
    </Card>
  )
})
