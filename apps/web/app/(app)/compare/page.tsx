'use client'
export const dynamic = 'force-dynamic'

import { useEffect, useMemo, useState } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { modelController } from '@/lib/model-controller'
import { benchmarkController, type BenchmarkResult } from '@/lib/benchmark-controller'
import { useToastStore } from '@/lib/toast-store'
import ModelsCard from '@/components/compare/ModelsCard'
import ComparisonTableCard from '@/components/compare/ComparisonTableCard'
import SummaryCard from '@/components/compare/SummaryCard'
// @ts-ignore — file exists; CI Ubuntu tsc can't resolve this path
import OutputComparisonCard from '../../components/compare/OutputComparisonCard'
import dynamicNext from 'next/dynamic'

const VisualComparisonCard = dynamicNext(() => import('@/components/compare/VisualComparisonCard'), { ssr: false })

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
      } catch { addToast('Failed to load models', 'error')
      } finally { setLoading(false) }
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
      addToast(`Performance test failed for ${modelId}`, 'error')
    } finally { setRunning(prev => { const n = new Set(prev); n.delete(modelId); return n }) }
  }

  const runAll = async () => { for (const m of models) await runBenchmark(m.id) }

  const clearResult = (modelId: string) => setResults(prev => { const n = { ...prev }; delete n[modelId]; return n })

  const completedResults = useMemo(() => Object.entries(results).filter(([, r]) => r !== null && !r!.error) as [string, BenchmarkResult][], [results])

  const bestMetrics: Record<string, number> = useMemo(() => {
    if (completedResults.length === 0) return { throughput: 0, latency: Infinity, p95: Infinity, params: 0 }
    return {
      throughput: Math.max(...completedResults.map(([, r]) => r.throughput_tokens_per_sec)),
      latency: Math.min(...completedResults.map(([, r]) => r.inference_time_ms)),
      p95: Math.min(...completedResults.map(([, r]) => r.latency_p95_ms ?? Infinity)),
      params: Math.max(...completedResults.map(([, r]) => r.num_parameters)),
    }
  }, [completedResults])

  const chartData = useMemo(() => completedResults
    .map(([modelId, r]) => ({ name: models.find(m => m.id === modelId)?.name || modelId, throughput: r.throughput_tokens_per_sec, latency: r.inference_time_ms, memory: r.memory_mb }))
    .sort((a, b) => b.throughput - a.throughput), [completedResults, models])

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Model Comparison" subtitle="Side-by-side benchmark results across models" />}
        right={<Button variant="outline" size="sm" onClick={runAll} disabled={loading || running.size > 0}><IconRefresh className="h-3.5 w-3.5 mr-1" /> Benchmark all</Button>}
      />

      <div className="space-y-4">
        <ModelsCard models={models} loading={loading} results={results} running={running} onBenchmark={runBenchmark} onClear={clearResult} />
        <ComparisonTableCard completedResults={completedResults} models={models} bestMetrics={bestMetrics} />
        <SummaryCard completedResults={completedResults} models={models} />
        <OutputComparisonCard models={models} />
        <VisualComparisonCard chartData={chartData} />
      </div>
    </div>
  )
}
