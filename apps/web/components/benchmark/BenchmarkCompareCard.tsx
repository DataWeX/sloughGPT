'use client'

import { useMemo } from 'react'
import { cn, Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface BenchmarkResult {
  model?: string
  throughput_tokens_per_sec?: number
  memory_mb?: number
  inference_time_ms?: number
  inference_count?: number
  total_tokens?: number
}

interface BenchmarkCompareCardProps {
  results: [string, BenchmarkResult][]
}

interface MetricDef {
  key: string
  label: string
  unit: string
  higherBetter: boolean
}

const METRICS: MetricDef[] = [
  { key: 'throughput_tokens_per_sec', label: 'Throughput', unit: 'tok/s', higherBetter: true },
  { key: 'memory_mb', label: 'Memory', unit: 'MB', higherBetter: false },
  { key: 'inference_time_ms', label: 'Latency', unit: 'ms', higherBetter: false },
  { key: 'inference_count', label: 'Inferences', unit: '', higherBetter: true },
  { key: 'total_tokens', label: 'Total Tokens', unit: '', higherBetter: true },
]

function getVal(result: BenchmarkResult, key: string): number {
  const v = (result as Record<string, unknown>)[key]
  if (typeof v === 'number') return v
  if (typeof v === 'object' && v !== null && 'value' in (v as Record<string, unknown>)) {
    return Number((v as { value: unknown }).value) || 0
  }
  return 0
}

export function BenchmarkCompareCard({ results }: BenchmarkCompareCardProps) {
  const bestValues = useMemo(() => {
    const best: Record<string, { value: number; model: string }> = {}
    for (const [model, result] of results) {
      for (const m of METRICS) {
        const val = getVal(result, m.key)
        if (val === 0) continue
        if (!best[m.key] || (m.higherBetter ? val > best[m.key].value : val < best[m.key].value)) {
          best[m.key] = { value: val, model }
        }
      }
    }
    return best
  }, [results])

  if (results.length === 0) {
    return (
      <Card data-testid="benchmark-compare">
        <CardHeader><CardTitle className="text-base">Model Comparison</CardTitle></CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground text-center py-3">Run benchmarks on multiple models to compare.</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card data-testid="benchmark-compare">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Model Comparison</CardTitle>
          <span className="text-[10px] text-muted-foreground">{results.length} model{results.length !== 1 ? 's' : ''}</span>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {METRICS.map(metric => {
          const values = results.map(([model, r]) => ({
            model,
            value: getVal(r, metric.key),
          })).filter(v => v.value > 0)

          if (values.length === 0) return null

          const maxVal = Math.max(...values.map(v => v.value))

          return (
            <div key={metric.key}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">{metric.label}</span>
                <span className="text-[9px] text-muted-foreground/60">{metric.unit}</span>
              </div>
              <div className="space-y-1">
                {values.map(v => {
                  const isBest = bestValues[metric.key]?.model === v.model
                  const pct = maxVal > 0 ? (v.value / maxVal) * 100 : 0
                  return (
                    <div key={v.model} className="flex items-center gap-2">
                      <span className={cn('text-[9px] w-20 truncate', isBest ? 'text-success font-medium' : 'text-muted-foreground')}>
                        {v.model}{isBest ? ' ★' : ''}
                      </span>
                      <div className="flex-1 h-4 bg-muted/30 rounded-sm overflow-hidden">
                        <div
                          className={cn('h-full rounded-sm transition-all duration-500', isBest ? 'bg-success/60' : 'bg-primary/30')}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="text-[9px] font-mono text-muted-foreground w-16 text-right">
                        {v.value >= 1000 ? `${(v.value / 1000).toFixed(1)}k` : v.value.toFixed(v.value < 10 ? 2 : 0)}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}

        <div className="pt-2 border-t border-border/30">
          <div className="grid grid-cols-2 gap-2">
            {results.map(([model, r]) => (
              <div key={model} className="rounded-md bg-muted/20 p-2 text-center">
                <p className="text-[10px] font-medium truncate">{model}</p>
                <p className="text-[9px] text-muted-foreground">
                  {r.total_tokens ?? 0} tokens · {r.inference_count ?? 0} runs
                </p>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
