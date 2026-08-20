'use client'
export const dynamic = 'force-dynamic'

import { useRouter } from 'next/navigation'
import { useEffect, useMemo, useState } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Button } from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { IconRefresh, IconDownload } from '@sloughgpt/strui'
import { modelController } from '@/lib/model-controller'
import { benchmarkController, type BenchmarkResult } from '@/lib/benchmark-controller'
import { useToastStore } from '@/lib/toast-store'
import { downloadJson } from '@/lib/download-utils'
import { todayDateString, getJsonItem } from '@/lib/format-bytes'
import ModelsCard from '@/components/compare/ModelsCard'
import ComparisonTableCard from '@/components/compare/ComparisonTableCard'
import SummaryCard from '@/components/compare/SummaryCard'
import { ModelComparisonInsightsCard } from '@/components/compare/ModelComparisonInsightsCard'
import dynamicNext from 'next/dynamic'

const OutputComparisonCard = dynamicNext<{ models: ModelEntry[] }>(() => import('@/components/compare/OutputComparisonCard') as Promise<{ default: React.ComponentType<{ models: ModelEntry[] }> }>, { ssr: false })

const VisualComparisonCard = dynamicNext(() => import('@/components/compare/VisualComparisonCard'), { ssr: false })

interface ModelEntry {
  id: string
  name: string
  loaded: boolean
  sizeGb?: number
}

interface SavedSnapshot {
  id: string
  name: string
  savedAt: string
  results: Record<string, BenchmarkResult>
  modelNames: Record<string, string>
}

const STORAGE_KEY = 'compare-snapshots'

function loadSnapshots(): SavedSnapshot[] {
  return getJsonItem<SavedSnapshot[]>(STORAGE_KEY, [])
}

function saveSnapshots(snapshots: SavedSnapshot[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshots))
}

export default function ComparePage() {
  const router = useRouter()
  const [models, setModels] = useState<ModelEntry[]>([])
  const [results, setResults] = useState<Record<string, BenchmarkResult | null>>({})
  const [running, setRunning] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(true)
  const [snapshots, setSnapshots] = useState<SavedSnapshot[]>([])
  const [snapshotName, setSnapshotName] = useState('')
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => { setSnapshots(loadSnapshots()) }, [])

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
          source: m.source,
          type: m.type,
        }))
        setModels(entries)
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

  const exportResults = () => {
    const data = completedResults.map(([modelId, r]) => ({
      model: models.find(m => m.id === modelId)?.name || modelId,
      throughput_tokens_per_sec: r.throughput_tokens_per_sec,
      inference_time_ms: r.inference_time_ms,
      latency_p95_ms: r.latency_p95_ms,
      memory_mb: r.memory_mb,
      num_parameters: r.num_parameters,
    }))
    downloadJson(data, `benchmark-comparison-${todayDateString()}.json`)
    addToast(`Exported ${data.length} results`, 'success')
  }

  const saveSnapshot = () => {
    if (completedResults.length === 0) return addToast('No results to save', 'error')
    const name = snapshotName.trim() || `Comparison ${new Date().toLocaleDateString()}`
    const modelNames: Record<string, string> = {}
    completedResults.forEach(([id]) => { modelNames[id] = models.find(m => m.id === id)?.name || id })
    const snap: SavedSnapshot = {
      id: Date.now().toString(36),
      name,
      savedAt: new Date().toISOString(),
      results: Object.fromEntries(completedResults),
      modelNames,
    }
    const updated = [snap, ...snapshots]
    setSnapshots(updated)
    saveSnapshots(updated)
    setSnapshotName('')
    addToast(`Saved "${name}"`, 'success')
  }

  const loadSnapshot = (snap: SavedSnapshot) => {
    setResults(snap.results)
    addToast(`Loaded "${snap.name}"`, 'success')
  }

  const deleteSnapshot = (id: string) => {
    const updated = snapshots.filter(s => s.id !== id)
    setSnapshots(updated)
    saveSnapshots(updated)
    addToast('Snapshot deleted', 'success')
  }

  const completedResults = useMemo(() => Object.entries(results).filter(([, r]) => r !== null && !r!.error) as [string, BenchmarkResult][], [results])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        runAll()
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault()
        if (completedResults.length > 0) saveSnapshot()
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault()
        if (completedResults.length > 0) exportResults()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [completedResults]) // eslint-disable-line react-hooks/exhaustive-deps

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

  const headerRight = (
    <div className="flex items-center gap-2">
      {completedResults.length > 0 && (
        <>
          <div className="flex items-center gap-1">
            <Input
              value={snapshotName}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSnapshotName(e.target.value)}
              placeholder="Snapshot name..."
              aria-label="Snapshot name"
              className="h-8 w-40 text-xs"
              onKeyDown={(e: React.KeyboardEvent) => { if (e.key === 'Enter') saveSnapshot() }}
            />
            <Button variant="outline" size="sm" onClick={saveSnapshot}>Save</Button>
          </div>
          <Button variant="outline" size="sm" onClick={exportResults}>
            <IconDownload className="h-3.5 w-3.5 mr-1" />
            Export
          </Button>
        </>
      )}
      <Button variant="outline" size="sm" onClick={runAll} disabled={loading || running.size > 0}>
        <IconRefresh className="h-3.5 w-3.5 mr-1" /> Benchmark all
      </Button>
    </div>
  )

  return (
    <PageContainer
      title="Model Comparison"
      subtitle="Side-by-side benchmark results across models"
      headerRight={headerRight}
    >
      {snapshots.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Saved Comparisons</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {snapshots.map(snap => (
                  <div key={snap.id} className="flex items-center gap-1 rounded-lg border border-border/40 bg-muted/20 px-2 py-1">
                    <button onClick={() => loadSnapshot(snap)} className="text-xs font-medium hover:text-primary transition-colors">
                      {snap.name}
                    </button>
                    <span className="text-[10px] text-muted-foreground">{new Date(snap.savedAt).toLocaleDateString()}</span>
                    <button onClick={() => deleteSnapshot(snap.id)} aria-label={`Delete snapshot ${snap.name}`} className="text-[10px] text-muted-foreground hover:text-destructive ml-1">×</button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        <ModelsCard models={models} loading={loading} results={results} running={running} onBenchmark={runBenchmark} onClear={clearResult} />
        {completedResults.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center space-y-3">
              <p className="text-sm text-muted-foreground">No benchmark results yet.</p>
              <p className="text-xs text-muted-foreground/70 max-w-md mx-auto">
                Run benchmarks on your models to see side-by-side comparisons. Click &ldquo;Benchmark all&rdquo; or use the benchmark button on each model card above.
              </p>
              <Button size="sm" variant="outline" className="h-8 text-xs" onClick={runAllBenchmarks} disabled={loading || models.length === 0}>
                Benchmark all
              </Button>
              <div className="flex items-center justify-center gap-4 text-[10px] text-muted-foreground/50 pt-2">
                <span><kbd className="px-1 py-0.5 rounded bg-muted/50 border border-border/50 font-mono">R</kbd> Benchmark all</span>
                <span><kbd className="px-1 py-0.5 rounded bg-muted/50 border border-border/50 font-mono">Ctrl+S</kbd> Save snapshot</span>
                <span><kbd className="px-1 py-0.5 rounded bg-muted/50 border border-border/50 font-mono">Ctrl+E</kbd> Export</span>
              </div>
            </CardContent>
          </Card>
        ) : (
          <>
            <ComparisonTableCard completedResults={completedResults} models={models} bestMetrics={bestMetrics} />
            <ModelComparisonInsightsCard completedResults={completedResults} models={models} bestMetrics={bestMetrics} />
            <SummaryCard completedResults={completedResults} models={models} />
            <OutputComparisonCard models={models} />
            <VisualComparisonCard chartData={chartData} />
          </>
        )}
    </PageContainer>
  )
}
