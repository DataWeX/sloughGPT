'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import type { BenchmarkResult } from '@/lib/benchmark-controller'

interface BenchmarkInsightsCardProps {
  metrics: BenchmarkResult | null
  quality: { coherence_score: number; quality_score: number; repetition_rate: number; total_responses?: number; avg_length?: number } | null
  stats: { total: number; avg_tokens: number } | null
}

function scoreLabel(score: number): { label: string; color: string } {
  if (score >= 0.8) return { label: 'Excellent', color: 'text-success' }
  if (score >= 0.6) return { label: 'Good', color: 'text-primary' }
  if (score >= 0.4) return { label: 'Fair', color: 'text-warning' }
  return { label: 'Poor', color: 'text-destructive' }
}

function extractMetric(metrics: BenchmarkResult, key: string): number | null {
  const val = (metrics as unknown as Record<string, unknown>)[key]
  if (typeof val === 'number') return val
  if (typeof val === 'object' && val !== null && 'value' in val) {
    return typeof (val as { value: unknown }).value === 'number' ? (val as { value: number }).value : null
  }
  return null
}

export function BenchmarkInsightsCard({ metrics, quality, stats }: BenchmarkInsightsCardProps) {
  if (!metrics && !quality && !stats) return null

  const qualityInfo = quality ? scoreLabel(quality.quality_score) : null
  const coherenceInfo = quality ? scoreLabel(quality.coherence_score) : null

  const perplexity = metrics ? extractMetric(metrics, 'perplexity') : null
  const throughput = metrics ? extractMetric(metrics, 'throughput_tokens_per_sec') : null
  const latency = metrics ? extractMetric(metrics, 'avg_latency_ms') : null

  return (
    <Card data-testid="benchmark-insights">
      <CardHeader>
        <CardTitle className="text-base">Performance Insights</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          {quality && (
            <div>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Quality</div>
              <div className={`text-lg font-semibold ${qualityInfo?.color}`}>{(quality.quality_score * 100).toFixed(0)}%</div>
              <div className="text-[10px] text-muted-foreground">{qualityInfo?.label}</div>
            </div>
          )}
          {quality && (
            <div>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Coherence</div>
              <div className={`text-lg font-semibold ${coherenceInfo?.color}`}>{(quality.coherence_score * 100).toFixed(0)}%</div>
              <div className="text-[10px] text-muted-foreground">{coherenceInfo?.label}</div>
            </div>
          )}
          {perplexity != null && (
            <div>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Perplexity</div>
              <div className="text-lg font-semibold">{perplexity.toFixed(2)}</div>
              <div className="text-[10px] text-muted-foreground">{perplexity < 20 ? 'Low' : perplexity < 50 ? 'Medium' : 'High'}</div>
            </div>
          )}
          {throughput != null && (
            <div>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Throughput</div>
              <div className="text-lg font-semibold">{throughput.toFixed(1)}</div>
              <div className="text-[10px] text-muted-foreground">tokens/sec</div>
            </div>
          )}
        </div>
        <div className="space-y-1.5 text-[11px] text-muted-foreground">
          {quality && quality.repetition_rate > 0.3 && (
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-warning" />
              High repetition rate ({(quality.repetition_rate * 100).toFixed(0)}%) — consider adjusting temperature
            </div>
          )}
          {quality && quality.repetition_rate <= 0.3 && quality.repetition_rate > 0 && (
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-success" />
              Low repetition rate ({(quality.repetition_rate * 100).toFixed(0)}%)
            </div>
          )}
          {latency != null && latency > 5000 && (
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-warning" />
              High latency ({(latency / 1000).toFixed(1)}s) — consider smaller model
            </div>
          )}
          {stats && stats.total > 0 && (
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-primary" />
              {stats.total} responses logged · avg {stats.avg_tokens.toFixed(0)} tokens
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
