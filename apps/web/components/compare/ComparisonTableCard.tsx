'use client'

import { cn, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import type { BenchmarkResult } from '@/lib/benchmark-controller'
import { METRIC_COLUMNS } from '@/lib/compare-config'

interface ComparisonTableCardProps {
  completedResults: [string, BenchmarkResult][]
  models: { id: string; name: string }[]
  bestMetrics: Record<string, number>
}

export default function ComparisonTableCard({ completedResults, models, bestMetrics }: ComparisonTableCardProps) {
  if (completedResults.length === 0) return null

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Comparison</CardTitle>
          <p className="text-xs text-muted-foreground">{completedResults.length} model{completedResults.length !== 1 ? 's' : ''}</p>
        </div>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-xs" aria-label="Model comparison results">
          <thead>
            <tr className="border-b border-border/50">
              <th scope="col" className="text-left py-2 pr-4 font-medium text-muted-foreground">Model</th>
              {METRIC_COLUMNS.map(col => (
                <th scope="col" key={col.key} className="text-right py-2 px-3 font-medium text-muted-foreground whitespace-nowrap">{col.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {completedResults
              .sort(([, a], [, b]) => b.throughput_tokens_per_sec - a.throughput_tokens_per_sec)
              .map(([modelId, r]) => {
                const modelName = models.find(m => m.id === modelId)?.name || modelId
                return (
                  <tr key={modelId} className="border-b border-border/20 hover:bg-muted/20 transition-colors">
                    <td className="py-2.5 pr-4 font-medium text-sm truncate max-w-[160px]">{modelName}</td>
                    {METRIC_COLUMNS.map(col => {
                      const val = col.accessor(r)
                      const metricKey = col.key === 'inference_time_ms' ? 'latency' : col.key === 'latency_p95_ms' ? 'p95' : col.key === 'throughput_tokens_per_sec' ? 'throughput' : col.key === 'num_parameters' ? 'params' : col.key
                      const isBest = col.lowerBetter ? val <= (bestMetrics[metricKey] ?? Infinity) : val >= (bestMetrics[metricKey] ?? -Infinity)
                      const isFiniteVal = val !== Infinity && val !== -Infinity
                      return (
                        <td key={col.key} className={cn("text-right py-2.5 px-3 whitespace-nowrap tabular-nums", isBest && isFiniteVal ? "text-success font-semibold" : "text-foreground/80")}>
                          {isFiniteVal ? col.fmt(val) : '—'}
                          {isBest && isFiniteVal && <span className="ml-1 text-[9px] text-success">●</span>}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
          </tbody>
        </table>
      </CardContent>
    </Card>
  )
}
