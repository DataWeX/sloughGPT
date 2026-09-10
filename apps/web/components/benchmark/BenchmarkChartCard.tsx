'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface BenchmarkHistoryEntry {
  timestamp: string
  model: string
  throughput?: number
  latency?: number
  memory?: number
  tokens?: number
  quality?: number
}

interface BenchmarkChartCardProps {
  history: BenchmarkHistoryEntry[]
}

type MetricKey = 'throughput' | 'latency' | 'memory' | 'tokens' | 'quality'

const METRICS: { key: MetricKey; label: string; color: string; unit: string }[] = [
  { key: 'throughput', label: 'Throughput', color: '#8b5cf6', unit: 'tok/s' },
  { key: 'latency', label: 'Latency', color: '#ef4444', unit: 'ms' },
  { key: 'memory', label: 'Memory', color: '#22c55e', unit: 'MB' },
  { key: 'tokens', label: 'Tokens', color: '#3b82f6', unit: '' },
  { key: 'quality', label: 'Quality', color: '#f59e0b', unit: '%' },
]

function SimpleLineChart({ data, dataKey, color, height = 60 }: { data: BenchmarkHistoryEntry[]; dataKey: string; color: string; height?: number }) {
  const points = useMemo(() => {
    const values = data.map(d => Number((d as unknown as Record<string, unknown>)[dataKey] ?? 0)).filter(v => !isNaN(v) && v > 0)
    if (values.length === 0) return ''
    const max = Math.max(...values)
    const min = Math.min(...values)
    const range = max - min || 1
    const step = 100 / Math.max(values.length - 1, 1)
    return values.map((v, i) => `${i * step},${100 - ((v - min) / range) * 100}`).join(' ')
  }, [data, dataKey])

  if (!points) return <div className="h-[60px] flex items-center justify-center text-[9px] text-muted-foreground">No data</div>

  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full" style={{ height }} role="img" aria-label={`${dataKey} chart`}>
      <polyline fill="none" stroke={color} strokeWidth="1.5" points={points} />
    </svg>
  )
}

function formatTime(ts: string): string {
  try {
    const d = new Date(ts)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return ts
  }
}

export function BenchmarkChartCard({ history }: BenchmarkChartCardProps) {
  const byModel = useMemo(() => {
    const map: Record<string, BenchmarkHistoryEntry[]> = {}
    for (const entry of history) {
      const model = entry.model ?? 'unknown'
      if (!map[model]) map[model] = []
      map[model].push(entry)
    }
    return map
  }, [history])

  const models = Object.keys(byModel)

  if (history.length === 0) {
    return (
      <Card data-testid="benchmark-chart">
        <CardHeader><CardTitle className="text-base">Metrics Over Time</CardTitle></CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground text-center py-3">No benchmark history yet.</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card data-testid="benchmark-chart">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Metrics Over Time</CardTitle>
          <span className="text-[10px] text-muted-foreground">{history.length} runs · {models.length} model{models.length !== 1 ? 's' : ''}</span>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {METRICS.map(metric => {
          const hasData = history.some(h => h[metric.key] != null && Number(h[metric.key]) > 0)
          if (!hasData) return null

          return (
            <div key={metric.key}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">{metric.label}</span>
                <span className="text-[9px] text-muted-foreground/60">{metric.unit}</span>
              </div>
              <div className="rounded-md bg-muted/20 p-2">
                {models.map(model => (
                  <div key={model} className="flex items-center gap-2 mb-1 last:mb-0">
                    <span className="text-[9px] text-muted-foreground w-16 truncate">{model}</span>
                    <div className="flex-1">
                      <SimpleLineChart
                        data={byModel[model]}
                        dataKey={metric.key}
                        color={metric.color}
                        height={24}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )
        })}

        <div>
          <p className="text-[9px] text-muted-foreground/60 mb-1">Recent runs</p>
          <div className="flex gap-1 overflow-x-auto pb-1">
            {history.slice(-10).map((h, i) => (
              <div key={i} className="shrink-0 text-[9px] text-muted-foreground/60 border border-border/30 rounded px-1.5 py-0.5">
                {h.model} · {formatTime(h.timestamp)}
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
