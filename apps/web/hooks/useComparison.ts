'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { modelController } from '@/lib/model-controller'
import { benchmarkController, type BenchmarkResult } from '@/lib/benchmark-controller'
import { useToastStore } from '@/lib/toast-store'
import { downloadJson } from '@/lib/download-utils'
import { todayDateString, getJsonItem, setJsonItem } from '@/lib/format-bytes'

export interface ModelEntry {
  id: string
  name: string
  loaded: boolean
  sizeGb?: number
  source?: string
  type?: string
}

export interface SavedSnapshot {
  id: string
  name: string
  savedAt: string
  results: Record<string, BenchmarkResult>
  modelNames: Record<string, string>
}

const STORAGE_KEY = 'compare-snapshots'

async function loadSnapshots(): Promise<SavedSnapshot[]> {
  return getJsonItem<SavedSnapshot[]>(STORAGE_KEY, [])
}

async function persistSnapshots(snapshots: SavedSnapshot[]) {
  await setJsonItem(STORAGE_KEY, snapshots)
}

export function useComparison() {
  const [models, setModels] = useState<ModelEntry[]>([])
  const [results, setResults] = useState<Record<string, BenchmarkResult | null>>({})
  const [running, setRunning] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(true)
  const [snapshots, setSnapshots] = useState<SavedSnapshot[]>([])
  const [snapshotName, setSnapshotName] = useState('')
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => {
    loadSnapshots().then(setSnapshots)
  }, [])

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
      } catch {
        addToast('Could not load models', 'error')
      } finally {
        setLoading(false)
      }
    })()
  }, [addToast])

  const runBenchmark = useCallback(async (modelId: string) => {
    setRunning(prev => new Set(prev).add(modelId))
    setResults(prev => ({ ...prev, [modelId]: null }))
    try {
      const result = await benchmarkController.run({ model: modelId })
      setResults(prev => ({ ...prev, [modelId]: result }))
    } catch {
      setResults(prev => ({ ...prev, [modelId]: { error: 'Failed' } as BenchmarkResult }))
      addToast(`Could not complete performance test for ${modelId}`, 'error')
    } finally {
      setRunning(prev => { const n = new Set(prev); n.delete(modelId); return n })
    }
  }, [addToast])

  const runAll = useCallback(async () => {
    for (const m of models) await runBenchmark(m.id)
  }, [models, runBenchmark])

  const clearResult = useCallback((modelId: string) => {
    setResults(prev => { const n = { ...prev }; delete n[modelId]; return n })
  }, [])

  const completedResults = useMemo(
    () => Object.entries(results).filter(([, r]) => r !== null && !r!.error) as [string, BenchmarkResult][],
    [results],
  )

  const exportResults = useCallback(() => {
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
  }, [completedResults, models, addToast])

  const saveSnapshot = useCallback(() => {
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
    persistSnapshots(updated).catch(() => {})
    setSnapshotName('')
    addToast(`Saved "${name}"`, 'success')
  }, [completedResults, snapshots, snapshotName, models, addToast])

  const loadSnapshot = useCallback((snap: SavedSnapshot) => {
    setResults(snap.results)
    addToast(`Loaded "${snap.name}"`, 'success')
  }, [addToast])

  const deleteSnapshot = useCallback((id: string) => {
    const updated = snapshots.filter(s => s.id !== id)
    setSnapshots(updated)
    persistSnapshots(updated).catch(() => {})
    addToast('Snapshot deleted', 'success')
  }, [snapshots, addToast])

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
    .map(([modelId, r]) => ({
      name: models.find(m => m.id === modelId)?.name || modelId,
      throughput: r.throughput_tokens_per_sec,
      latency: r.inference_time_ms,
      memory: r.memory_mb,
    }))
    .sort((a, b) => b.throughput - a.throughput), [completedResults, models])

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
  }, [completedResults, runAll, saveSnapshot, exportResults])

  return {
    models,
    results,
    running,
    loading,
    snapshots,
    snapshotName,
    setSnapshotName,
    completedResults,
    bestMetrics,
    chartData,
    runBenchmark,
    runAll,
    clearResult,
    exportResults,
    saveSnapshot,
    loadSnapshot,
    deleteSnapshot,
  }
}
