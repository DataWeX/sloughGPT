'use client'

import { memo } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface BenchmarkResult {
  throughput_tokens_per_sec: number
  inference_time_ms: number
  latency_p95_ms?: number
  memory_mb: number
  num_parameters: number
  error?: string
}

interface ModelComparisonInsightsCardProps {
  completedResults: [string, BenchmarkResult][]
  models: Array<{ id: string; name: string }>
  bestMetrics: Record<string, number>
}

function winner(modelId: string, results: [string, BenchmarkResult][], key: keyof BenchmarkResult, lower: boolean): boolean {
  const vals = results.filter(([, r]) => !r.error).map(([, r]) => r[key] as number)
  if (vals.length === 0) return false
  const best = lower ? Math.min(...vals) : Math.max(...vals)
  const entry = results.find(([id]) => id === modelId)
  return entry ? (entry[1][key] as number) === best : false
}

export const ModelComparisonInsightsCard = memo(function ModelComparisonInsightsCard({ completedResults, models, bestMetrics }: ModelComparisonInsightsCardProps) {
  if (completedResults.length < 2) return null

  const fastest = completedResults
    .filter(([, r]) => !r.error)
    .sort((a, b) => a[1].inference_time_ms - b[1].inference_time_ms)[0]
  const mostThroughput = completedResults
    .filter(([, r]) => !r.error)
    .sort((a, b) => b[1].throughput_tokens_per_sec - a[1].throughput_tokens_per_sec)[0]
  const leastMemory = completedResults
    .filter(([, r]) => !r.error)
    .sort((a, b) => a[1].memory_mb - b[1].memory_mb)[0]

  const fastestName = fastest ? (models.find(m => m.id === fastest[0])?.name || fastest[0]) : '—'
  const throughputName = mostThroughput ? (models.find(m => m.id === mostThroughput[0])?.name || mostThroughput[0]) : '—'
  const memoryName = leastMemory ? (models.find(m => m.id === leastMemory[0])?.name || leastMemory[0]) : '—'

  const latencySpread = bestMetrics.latency && bestMetrics.latency !== Infinity
    ? ((bestMetrics.latency - Math.min(...completedResults.filter(([, r]) => !r.error).map(([, r]) => r.inference_time_ms))) / bestMetrics.latency * 100)
    : 0

  return (
    <Card data-testid="model-comparison-insights">
      <CardHeader>
        <CardTitle className="text-base">Comparison Insights</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Fastest</div>
            <div className="text-sm font-semibold truncate">{fastestName}</div>
            {fastest && (
              <div className="text-[10px] text-muted-foreground">{(fastest[1].inference_time_ms / 1000).toFixed(2)}s</div>
            )}
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Highest Throughput</div>
            <div className="text-sm font-semibold truncate">{throughputName}</div>
            {mostThroughput && (
              <div className="text-[10px] text-muted-foreground">{mostThroughput[1].throughput_tokens_per_sec.toFixed(1)} tok/s</div>
            )}
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Most Efficient</div>
            <div className="text-sm font-semibold truncate">{memoryName}</div>
            {leastMemory && (
              <div className="text-[10px] text-muted-foreground">{leastMemory[1].memory_mb.toFixed(0)} MB</div>
            )}
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Models Compared</div>
            <div className="text-sm font-semibold">{completedResults.length}</div>
            <div className="text-[10px] text-muted-foreground">
              {latencySpread > 30 ? 'Wide gap' : latencySpread > 10 ? 'Moderate gap' : 'Close match'}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
})
