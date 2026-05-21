'use client'

import { useMemo, useState } from 'react'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { benchmarkController, type BenchmarkResult } from '@/lib/controllers'
import { useApiHealth } from '@/hooks/useApiHealth'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { StatCard, KpiGrid, Skeleton } from '@/components/ui/display'
import { Badge } from '@/components/ui/tags'

export default function BenchmarkPage() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<BenchmarkResult | null>(null)
  const [prompt, setPrompt] = useState('The quick brown fox jumps over the lazy dog')
  const [error, setError] = useState<string | null>(null)
  const { state: apiHealth } = useApiHealth()

  const canRunBenchmark = useMemo(() => {
    if (apiHealth === null) return false
    if (apiHealth === 'offline') return false
    return apiHealth.model_loaded
  }, [apiHealth])

  const statusLabel = useMemo(() => {
    if (apiHealth === null) return 'Checking…'
    if (apiHealth === 'offline') return 'API unreachable'
    if (!apiHealth.model_loaded) return 'No model loaded'
    return apiHealth.model_type
  }, [apiHealth])

  const statusVariant = useMemo(() => {
    if (apiHealth === null) return 'warning' as const
    if (apiHealth === 'offline') return 'error' as const
    if (!apiHealth.model_loaded) return 'warning' as const
    return 'success' as const
  }, [apiHealth])

  const runBenchmark = async () => {
    setLoading(true); setError(null)
    try {
      const res = await benchmarkController.run({ dataset: prompt })
      setResult(res)
    } catch { setError('Failed to run benchmark') }
    finally { setLoading(false) }
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        className="items-start"
        left={
          <AppRouteHeaderLead
            title="Benchmark"
            subtitle={
              <span className="flex items-center gap-2">
                Model:
                <Badge label={statusLabel} variant={statusVariant} size="sm" />
              </span>
            }
          />
        }
      />

      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Run inference benchmark</CardTitle>
            <CardDescription>Measure model throughput and latency.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="bench-prompt">Prompt</Label>
              <Input id="bench-prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            </div>

            {!canRunBenchmark && (
              <p className="text-sm text-muted-foreground">
                Load a model in the Models page before running a benchmark.
              </p>
            )}

            <Button type="button" onClick={runBenchmark} disabled={loading || !canRunBenchmark}>
              {loading ? 'Running…' : 'Run benchmark'}
            </Button>
            {error && <p className="text-sm text-destructive">{error}</p>}
          </CardContent>
        </Card>

        {loading && (
          <KpiGrid columns={4}>
            {[1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-24 rounded-lg" />
            ))}
          </KpiGrid>
        )}

        {result && !result.error && (
          <>
            <KpiGrid columns={4}>
              <StatCard label="Parameters" value={result.num_parameters.toLocaleString()} />
              <StatCard label="Memory" value={`${result.memory_mb.toFixed(1)} MB`} />
              <StatCard label="Throughput" value={`${result.throughput_tokens_per_sec.toFixed(2)} tok/s`} />
              <StatCard label="Avg latency" value={`${result.inference_time_ms.toFixed(0)} ms`} />
            </KpiGrid>
            <KpiGrid columns={3}>
              <StatCard label="P50" value={`${result.latency_p50_ms?.toFixed(0) ?? '—'} ms`} />
              <StatCard label="P95" value={`${result.latency_p95_ms?.toFixed(0) ?? '—'} ms`} />
              <StatCard label="P99" value={`${result.latency_p99_ms?.toFixed(0) ?? '—'} ms`} />
            </KpiGrid>
          </>
        )}
      </div>
    </div>
  )
}
