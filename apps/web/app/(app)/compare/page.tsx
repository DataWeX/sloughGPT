'use client'

import { useEffect, useMemo, useState } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/tags'
import { StatCard, KpiGrid, Skeleton } from '@/components/ui/display'
import { IconRefresh, IconTrash, IconCheck } from '@/components/ui'
import { cn } from '@/lib/cn'
import { modelController } from '@/lib/model-controller'
import { benchmarkController, type BenchmarkResult } from '@/lib/benchmark-controller'
import { useToastStore } from '@/lib/toast-store'

interface ModelEntry {
  id: string
  name: string
  loaded: boolean
  sizeGb?: number
}

export default function ComparePage() {
  const [models, setModels] = useState<ModelEntry[]>([])
  const [results, setResults] = useState<Record<string, BenchmarkResult | null>>({})
  const [running, setRunning] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(true)
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => {
    (async () => {
      try {
        const list = await modelController.list()
        const health = await modelController.getHealth()
        const entries: ModelEntry[] = list.map(m => ({
          id: m.id || m.name,
          name: (m.id || m.name).replace(/^hf\//, ''),
          loaded: m.loaded || (health?.model_type?.includes(m.id || m.name) ?? false),
          sizeGb: m.size_gb,
        }))
        setModels(entries.slice(0, 8))
      } catch {
        addToast('Failed to fetch models', 'error')
      } finally {
        setLoading(false)
      }
    })()
  }, [addToast])

  const runBenchmark = async (modelId: string) => {
    setRunning(prev => new Set(prev).add(modelId))
    setResults(prev => ({ ...prev, [modelId]: null }))
    try {
      const result = await benchmarkController.run({ model: modelId })
      setResults(prev => ({ ...prev, [modelId]: result }))
    } catch {
      setResults(prev => ({ ...prev, [modelId]: { error: 'Failed' } as BenchmarkResult }))
      addToast(`Benchmark failed for ${modelId}`, 'error')
    } finally {
      setRunning(prev => { const n = new Set(prev); n.delete(modelId); return n })
    }
  }

  const runAll = async () => {
    for (const m of models) {
      await runBenchmark(m.id)
    }
  }

  const clearResult = (modelId: string) => {
    setResults(prev => {
      const n = { ...prev }
      delete n[modelId]
      return n
    })
  }

  const completedResults = useMemo(() => {
    return Object.entries(results).filter(
      ([, r]) => r !== null && !r!.error
    ) as [string, BenchmarkResult][]
  }, [results])

  const bestMetrics = useMemo(() => {
    if (completedResults.length === 0) return {}
    return {
      throughput: Math.max(...completedResults.map(([, r]) => r.throughput_tokens_per_sec)),
      latency: Math.min(...completedResults.map(([, r]) => r.inference_time_ms)),
      p95: Math.min(...completedResults.map(([, r]) => r.latency_p95_ms ?? Infinity)),
      params: Math.max(...completedResults.map(([, r]) => r.num_parameters)),
    }
  }, [completedResults])

  const metricColumns: {
    label: string
    key: keyof BenchmarkResult
    fmt: (v: number) => string
    lowerBetter?: boolean
    accessor: (r: BenchmarkResult) => number
  }[] = [
    { label: 'Parameters', key: 'num_parameters', fmt: v => v.toLocaleString(), accessor: r => r.num_parameters },
    { label: 'Memory', key: 'memory_mb', fmt: v => `${v.toFixed(0)} MB`, accessor: r => r.memory_mb },
    { label: 'Throughput', key: 'throughput_tokens_per_sec', fmt: v => `${v.toFixed(1)} tok/s`, accessor: r => r.throughput_tokens_per_sec },
    { label: 'Avg latency', key: 'inference_time_ms', fmt: v => `${v.toFixed(0)} ms`, lowerBetter: true, accessor: r => r.inference_time_ms },
    { label: 'P95', key: 'latency_p95_ms', fmt: v => `${v.toFixed(0)} ms`, lowerBetter: true, accessor: r => r.latency_p95_ms ?? Infinity },
    { label: 'P99', key: 'latency_p99_ms', fmt: v => `${v.toFixed(0)} ms`, lowerBetter: true, accessor: r => r.latency_p99_ms ?? Infinity },
  ]

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={
          <AppRouteHeaderLead
            title="Model Comparison"
            subtitle="Side-by-side benchmark results across models"
          />
        }
        right={
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={runAll}
              disabled={loading || running.size > 0}
            >
              <IconRefresh className="h-3.5 w-3.5 mr-1" />
              Benchmark all
            </Button>
          </div>
        }
      />

      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Models</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-28 rounded-lg" />
                ))}
              </div>
            ) : models.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-6">
                No models available. Load one in the Models page first.
              </p>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                {models.map(m => {
                  const result = results[m.id]
                  const isRunning = running.has(m.id)
                  return (
                    <div
                      key={m.id}
                      className={cn(
                        "rounded-lg border p-3 transition-all",
                        result && !result.error
                          ? "border-primary/30 bg-primary/5"
                          : m.loaded
                            ? "border-border/60 bg-card/50"
                            : "border-border/30 bg-muted/20 opacity-70"
                      )}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm font-medium truncate">{m.name}</p>
                        {m.loaded && (
                          <Badge label="Loaded" variant="success" size="sm" />
                        )}
                      </div>
                      {m.sizeGb && (
                        <p className="text-xs text-muted-foreground mb-2">{m.sizeGb.toFixed(1)} GB</p>
                      )}
                      <div className="flex items-center gap-1.5">
                        <Button
                          size="sm"
                          variant={result ? "outline" : "default"}
                          className="h-7 text-xs flex-1"
                          onClick={() => runBenchmark(m.id)}
                          disabled={isRunning}
                        >
                          {isRunning ? (
                            <>Benchmarking…</>
                          ) : result ? (
                            <><IconCheck className="h-3 w-3 mr-1" /> Rerun</>
                          ) : (
                            'Benchmark'
                          )}
                        </Button>
                        {result && (
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            className="h-7 w-7"
                            onClick={() => clearResult(m.id)}
                            aria-label={`Clear result for ${m.id}`}
                          >
                            <IconTrash className="h-3 w-3" />
                          </Button>
                        )}
                      </div>
                      {result?.error && (
                        <p className="text-[10px] text-destructive mt-1">{result.error}</p>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </CardContent>
        </Card>

        {completedResults.length > 0 && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Comparison</CardTitle>
                <p className="text-xs text-muted-foreground">
                  {completedResults.length} model{completedResults.length !== 1 ? 's' : ''}
                </p>
              </div>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border/50">
                    <th className="text-left py-2 pr-4 font-medium text-muted-foreground">Model</th>
                    {metricColumns.map(col => (
                      <th key={col.key} className="text-right py-2 px-3 font-medium text-muted-foreground whitespace-nowrap">
                        {col.label}
                      </th>
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
                          {metricColumns.map(col => {
                            const val = col.accessor(r)
                            const isBest = col.lowerBetter
                              ? val <= (bestMetrics as any)[col.key === 'inference_time_ms' ? 'latency' : col.key === 'latency_p95_ms' ? 'p95' : col.key]
                              : val >= (bestMetrics as any)[col.key === 'throughput_tokens_per_sec' ? 'throughput' : col.key === 'num_parameters' ? 'params' : col.key]
                            const isFinite = val !== Infinity && val !== -Infinity
                            return (
                              <td
                                key={col.key}
                                className={cn(
                                  "text-right py-2.5 px-3 whitespace-nowrap tabular-nums",
                                  isBest && isFinite ? "text-success font-semibold" : "text-foreground/80"
                                )}
                              >
                                {isFinite ? col.fmt(val) : '—'}
                                {isBest && isFinite && (
                                  <span className="ml-1 text-[9px] text-success">●</span>
                                )}
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
        )}

        {completedResults.length > 1 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <KpiGrid columns={completedResults.length >= 4 ? 4 : 2}>
                {completedResults.map(([modelId, r]) => {
                  const modelName = models.find(m => m.id === modelId)?.name || modelId
                  return (
                    <StatCard
                      key={modelId}
                      label={modelName}
                      value={`${r.throughput_tokens_per_sec.toFixed(1)} tok/s`}
                    />
                  )
                })}
              </KpiGrid>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
