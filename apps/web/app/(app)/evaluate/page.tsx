'use client'
export const dynamic = 'force-dynamic'

import { useRouter } from 'next/navigation'
import { useEffect, useMemo, useState } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Button, cn } from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Input, Textarea } from '@sloughgpt/strui'
import { IconRefresh, IconDownload } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { modelController } from '@/lib/model-controller'
import { benchmarkController, type BenchmarkResult, type LoggedBenchmarkResponse } from '@/lib/benchmark-controller'
import { apiPost } from '@/lib/http-client'
import { BenchmarkInsightsCard } from '@/components/benchmark/BenchmarkInsightsCard'
import { useToastStore } from '@/lib/toast-store'
import { downloadJson } from '@/lib/download-utils'
import { todayDateString, getJsonItem } from '@/lib/format-bytes'
import ModelsCard from '@/components/compare/ModelsCard'
import ComparisonTableCard from '@/components/compare/ComparisonTableCard'
import SummaryCard from '@/components/compare/SummaryCard'
import { ModelComparisonInsightsCard } from '@/components/compare/ModelComparisonInsightsCard'
import dynamicNext from 'next/dynamic'

const OutputComparisonCard = dynamicNext<{ models: CompareModelEntry[] }>(() => import('@/components/compare/OutputComparisonCard') as Promise<{ default: React.ComponentType<{ models: CompareModelEntry[] }> }>, { ssr: false })
const VisualComparisonCard = dynamicNext(() => import('@/components/compare/VisualComparisonCard'), { ssr: false })

type SingleModelTab = 'metrics' | 'quality' | 'responses' | 'perplexity'

interface CompareModelEntry {
  id: string
  name: string
  loaded: boolean
  sizeGb?: number
  source?: string
  type?: string
}

interface SavedSnapshot {
  id: string
  name: string
  savedAt: string
  results: Record<string, BenchmarkResult>
  modelNames: Record<string, string>
}

const COMPARE_STORAGE_KEY = 'compare-snapshots'

function loadSnapshots(): SavedSnapshot[] {
  return getJsonItem<SavedSnapshot[]>(COMPARE_STORAGE_KEY, [])
}

function saveSnapshots(snapshots: SavedSnapshot[]) {
  localStorage.setItem(COMPARE_STORAGE_KEY, JSON.stringify(snapshots))
}

export default function EvaluatePage() {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const [section, setSection] = useState<'single' | 'compare'>('single')

  // ── Single Model state ──
  const [smTab, setSmTab] = useState<SingleModelTab>('metrics')
  const [metrics, setMetrics] = useState<BenchmarkResult | null>(null)
  const [quality, setQuality] = useState<{ coherence_score: number; quality_score: number; repetition_rate: number; total_responses: number; avg_length: number } | null>(null)
  const [responses, setResponses] = useState<LoggedBenchmarkResponse[]>([])
  const [stats, setStats] = useState<{ total: number; avg_tokens: number } | null>(null)
  const [smLoading, setSmLoading] = useState(true)
  const [smRunning, setSmRunning] = useState(false)
  const [smLoadError, setSmLoadError] = useState<string | null>(null)
  const [currentModel, setCurrentModel] = useState<string>('gpt2')
  const [pplxText, setPplxText] = useState('')
  const [pplxResult, setPplxResult] = useState<{ perplexity: number; loss: number; tokens: number } | null>(null)
  const [pplxLoading, setPplxLoading] = useState(false)

  // ── Comparison state ──
  const [cmpModels, setCmpModels] = useState<CompareModelEntry[]>([])
  const [cmpResults, setCmpResults] = useState<Record<string, BenchmarkResult | null>>({})
  const [cmpRunning, setCmpRunning] = useState<Set<string>>(new Set())
  const [cmpLoading, setCmpLoading] = useState(true)
  const [snapshots, setSnapshots] = useState<SavedSnapshot[]>([])
  const [snapshotName, setSnapshotName] = useState('')

  // ── Single Model effects ──
  const loadBenchmark = async () => {
    setSmLoading(true)
    setSmLoadError(null)
    let model = 'gpt2'
    try {
      const h = await modelController.getHealth()
      model = h?.model_type ?? 'gpt2'
      setCurrentModel(model)
    } catch { /* use default */ }

    try {
      const [m, q, s] = await Promise.all([
        benchmarkController.metrics(model).catch(() => null),
        benchmarkController.quality().catch(() => null),
        benchmarkController.stats().catch(() => null),
      ])
      setMetrics(m)
      setQuality(q)
      setStats(s)
      if (!m && !q && !s) setSmLoadError('Could not load benchmark data. Please try again.')
    } catch {
      setSmLoadError('Could not load benchmark data. Please try again.')
    } finally {
      setSmLoading(false)
    }
  }

  useEffect(() => { loadBenchmark() }, [])

  // ── Comparison effects ──
  useEffect(() => { setSnapshots(loadSnapshots()) }, [])

  useEffect(() => {
    (async () => {
      try {
        const list = await modelController.list()
        const health = await modelController.getHealth()
        const entries: CompareModelEntry[] = list.map(m => ({
          id: m.id || m.name,
          name: (m.id || m.name).replace(/^hf\//, ''),
          loaded: m.loaded || (health?.model_type?.includes(m.id || m.name) ?? false),
          sizeGb: m.size_gb,
          source: m.source,
          type: m.type,
        }))
        setCmpModels(entries)
      } catch { addToast('Could not load models', 'error')
      } finally { setCmpLoading(false) }
    })()
  }, [addToast])

  // ── Single Model handlers ──
  const handleRefreshMetrics = async () => {
    setSmRunning(true)
    try {
      const m = await benchmarkController.run({ model: currentModel })
      setMetrics(m)
    } catch {
      addToast('Could not run benchmark', 'error')
    } finally {
      setSmRunning(false)
    }
  }

  const handleLoadResponses = async () => {
    try {
      const data = await benchmarkController.history(20)
      setResponses(data)
    } catch {
      addToast('Could not load responses', 'error')
    }
  }

  const handleClearHistory = async () => {
    try {
      await apiPost('/benchmark/history/clear', {})
      setResponses([])
      setStats(null)
    } catch {
      addToast('Could not clear history', 'error')
    }
  }

  const handleCalcPerplexity = async () => {
    if (!pplxText.trim()) return
    setPplxLoading(true)
    try {
      const data = await apiPost<{ perplexity: number; loss: number; tokens: number }>(
        '/benchmark/perplexity',
        { text: pplxText },
      )
      setPplxResult(data)
    } catch {
      addToast('Could not calculate perplexity', 'error')
    } finally {
      setPplxLoading(false)
    }
  }

  // ── Comparison handlers ──
  const runBenchmark = async (modelId: string) => {
    setCmpRunning(prev => new Set(prev).add(modelId))
    setCmpResults(prev => ({ ...prev, [modelId]: null }))
    try {
      const result = await benchmarkController.run({ model: modelId })
      setCmpResults(prev => ({ ...prev, [modelId]: result }))
    } catch {
      setCmpResults(prev => ({ ...prev, [modelId]: { error: 'Failed' } as BenchmarkResult }))
      addToast(`Could not complete performance test for ${modelId}`, 'error')
    } finally { setCmpRunning(prev => { const n = new Set(prev); n.delete(modelId); return n }) }
  }

  const runAll = async () => { for (const m of cmpModels) await runBenchmark(m.id) }

  const clearResult = (modelId: string) => setCmpResults(prev => { const n = { ...prev }; delete n[modelId]; return n })

  const exportResults = () => {
    const data = completedResults.map(([modelId, r]) => ({
      model: cmpModels.find(m => m.id === modelId)?.name || modelId,
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
    completedResults.forEach(([id]) => { modelNames[id] = cmpModels.find(m => m.id === id)?.name || id })
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
    setCmpResults(snap.results)
    addToast(`Loaded "${snap.name}"`, 'success')
  }

  const deleteSnapshot = (id: string) => {
    const updated = snapshots.filter(s => s.id !== id)
    setSnapshots(updated)
    saveSnapshots(updated)
    addToast('Snapshot deleted', 'success')
  }

  const completedResults = useMemo(() => Object.entries(cmpResults).filter(([, r]) => r !== null && !r!.error) as [string, BenchmarkResult][], [cmpResults])

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
    .map(([modelId, r]) => ({ name: cmpModels.find(m => m.id === modelId)?.name || modelId, throughput: r.throughput_tokens_per_sec, latency: r.inference_time_ms, memory: r.memory_mb }))
    .sort((a, b) => b.throughput - a.throughput), [completedResults, cmpModels])

  // ── Shared header ──
  const headerRight = (
    <div className="flex items-center gap-2">
      {section === 'compare' && completedResults.length > 0 && (
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
      <Button variant="outline" size="sm" onClick={section === 'single' ? handleRefreshMetrics : runAll} disabled={section === 'single' ? smRunning : cmpLoading || cmpRunning.size > 0}>
        <IconRefresh className={cn('h-3.5 w-3.5 mr-1', (section === 'single' ? smRunning : cmpRunning.size > 0) && 'animate-spin')} />
        {section === 'single' ? 'Refresh' : 'Benchmark all'}
      </Button>
    </div>
  )

  const loading = section === 'single' ? smLoading : cmpLoading
  const error = section === 'single' ? smLoadError : null

  return (
    <PageContainer
      title="Evaluate"
      subtitle="Model evaluation and comparison"
      loading={loading}
      error={error}
      onRetry={() => { setSmLoading(true); loadBenchmark() }}
      headerRight={headerRight}
    >
      {/* Section tabs */}
      <div className="flex gap-1 border-b border-border/30 pb-0">
        {([
          { key: 'single' as const, label: 'Single Model' },
          { key: 'compare' as const, label: 'Comparison' },
        ]).map(t => (
          <button
            key={t.key}
            type="button"
            onClick={() => setSection(t.key)}
            className={cn('px-3 py-1.5 text-xs font-medium rounded-t transition-colors', section === t.key ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground hover:text-foreground')}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Single Model tab ── */}
      {section === 'single' && (
        <>
          <div className="flex gap-1 border-b border-border/30 pb-0">
            {(['metrics', 'quality', 'responses', 'perplexity'] as SingleModelTab[]).map(t => (
              <button
                key={t}
                type="button"
                onClick={() => {
                  setSmTab(t)
                  if (t === 'responses') handleLoadResponses()
                }}
                className={cn('px-3 py-1.5 text-xs font-medium rounded-t transition-colors', smTab === t ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground hover:text-foreground')}
              >
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>

          {smTab === 'metrics' && (
            <>
              <BenchmarkInsightsCard metrics={metrics} quality={quality} stats={stats} />
              <KpiGrid>
                <StatCard label="Model" value={String(metrics?.model ?? '—')} />
                <StatCard label="Responses" value={String(metrics?.inference_count ?? 0)} />
                <StatCard label="Speed" value={`${String(metrics?.tokens_per_second ?? 0)} tokens/s`} />
                <StatCard label="Memory" value={`${metrics?.memory_mb ?? 0} MB`} />
                <StatCard label="Total tokens" value={String(metrics?.total_tokens ?? 0)} />
                <StatCard label="Loaded" value={metrics?.model_loaded ? 'Yes' : 'No'} />
              </KpiGrid>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                  <CardTitle className="text-base">Model Metrics</CardTitle>
                  <Button size="sm" variant="ghost" onClick={handleRefreshMetrics} disabled={smRunning} aria-label="Refresh metrics">
                    <IconRefresh className={cn('h-4 w-4', smRunning && 'animate-spin')} />
                  </Button>
                </CardHeader>
                <CardContent>
                  {metrics ? (
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                      {[
                        { label: 'Model', value: String(metrics.model ?? '—') },
                        { label: 'Responses', value: String(metrics.inference_count ?? 0) },
                        { label: 'Total tokens', value: String(metrics.total_tokens ?? 0) },
                        { label: 'Speed', value: `${String(metrics.tokens_per_second ?? 0)} tokens/s` },
                        { label: 'Memory', value: `${metrics.memory_mb ?? 0} MB` },
                        { label: 'Loaded', value: metrics.model_loaded ? 'Yes' : 'No' },
                      ].map(s => (
                        <div key={s.label} className="rounded-md bg-muted/30 p-3 text-center">
                          <div className="text-xs text-muted-foreground">{s.label}</div>
                          <div className="text-sm font-mono font-medium">{s.value}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-6 text-sm text-muted-foreground">
                      No quality data yet. Chat with the model to generate responses.
                      <div className="mt-2">
                        <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => router.push('/models')}>
                          Open Models
                        </Button>
                      </div>
                    </div>
                  )}
                  {stats && (
                    <div className="mt-3 text-xs text-muted-foreground">
                      {stats.total} responses logged · avg {stats.avg_tokens?.toFixed(0) ?? 0} tokens
                    </div>
                  )}
                </CardContent>
              </Card>
            </>
          )}

          {smTab === 'quality' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Quality Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                {quality ? (
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {[
                      { label: 'Coherence', value: `${(quality.coherence_score * 100).toFixed(1)}%`, color: 'text-success' },
                      { label: 'Quality', value: `${(quality.quality_score * 100).toFixed(1)}%`, color: 'text-primary' },
                      { label: 'Repetition', value: `${(quality.repetition_rate * 100).toFixed(1)}%`, color: quality.repetition_rate > 0.3 ? 'text-destructive' : 'text-muted-foreground' },
                    ].map(s => (
                      <div key={s.label} className="rounded-md bg-muted/30 p-3 text-center">
                        <div className="text-xs text-muted-foreground">{s.label}</div>
                        <div className={cn('text-lg font-mono font-medium', s.color)}>{s.value}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-6 text-sm text-muted-foreground space-y-2">
                    <div>No quality data yet. Chat with the model to generate responses.</div>
                    <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => router.push('/chat')}>
                      Open Chat
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {smTab === 'responses' && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-base">Logged Responses ({responses.length})</CardTitle>
                <div className="flex gap-1">
                  <Button size="sm" variant="ghost" onClick={handleLoadResponses} aria-label="Refresh responses">
                    <IconRefresh className="h-4 w-4" />
                  </Button>
                  <Button size="sm" variant="ghost" className="text-destructive" onClick={handleClearHistory}>
                    Clear
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {responses.length === 0 ? (
                  <div className="text-center py-6 text-sm text-muted-foreground space-y-2">
                    <div>No responses logged yet.</div>
                    <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => router.push('/chat')}>
                      Open Chat
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {responses.map((r, i) => (
                      <div key={i} className="rounded-md border border-border/60 px-3 py-2 text-sm">
                        <div className="text-xs text-muted-foreground mb-1">
                          {r.timestamp ? new Date(r.timestamp).toLocaleString() : '—'} · {r.model} · {r.tokens_generated} tokens · {r.duration_ms?.toFixed(0)}ms
                        </div>
                        <div className="text-xs"><span className="text-muted-foreground">User:</span> {r.user_message}</div>
                        <div className="text-xs mt-0.5"><span className="text-muted-foreground">AI:</span> {r.assistant_response}</div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {smTab === 'perplexity' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Perplexity Calculator</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Textarea
                  value={pplxText}
                  onChange={e => setPplxText(e.target.value)}
                  placeholder="Enter text to calculate perplexity..."
                  rows={3}
                />
                <Button size="sm" onClick={handleCalcPerplexity} disabled={pplxLoading || !pplxText.trim()}>
                  {pplxLoading ? 'Calculating...' : 'Calculate'}
                </Button>
                {pplxResult && (
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="rounded-md bg-muted/30 p-3 text-center">
                      <div className="text-xs text-muted-foreground">Perplexity</div>
                      <div className="text-lg font-mono font-medium">{pplxResult.perplexity}</div>
                    </div>
                    <div className="rounded-md bg-muted/30 p-3 text-center">
                      <div className="text-xs text-muted-foreground">Loss</div>
                      <div className="text-lg font-mono font-medium">{pplxResult.loss}</div>
                    </div>
                    <div className="rounded-md bg-muted/30 p-3 text-center">
                      <div className="text-xs text-muted-foreground">Tokens</div>
                      <div className="text-lg font-mono font-medium">{pplxResult.tokens}</div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </>
      )}

      {/* ── Comparison tab ── */}
      {section === 'compare' && (
        <>
          {snapshots.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Saved Comparisons</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {snapshots.map(snap => (
                    <div key={snap.id} className="flex items-center gap-1 rounded-lg border border-border/40 bg-muted/20 px-2 py-1">
                      <button type="button" onClick={() => loadSnapshot(snap)} className="text-xs font-medium hover:text-primary transition-colors">
                        {snap.name}
                      </button>
                      <span className="text-[10px] text-muted-foreground">{new Date(snap.savedAt).toLocaleDateString()}</span>
                      <button type="button" onClick={() => deleteSnapshot(snap.id)} aria-label={`Delete snapshot ${snap.name}`} className="text-[10px] text-muted-foreground hover:text-destructive ml-1">×</button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          <ModelsCard models={cmpModels} loading={cmpLoading} results={cmpResults} running={cmpRunning} onBenchmark={runBenchmark} onClear={clearResult} />
          {completedResults.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center space-y-3">
                <p className="text-sm text-muted-foreground">No benchmark results yet.</p>
                <p className="text-xs text-muted-foreground/70 max-w-md mx-auto">
                  Run benchmarks on your models to see side-by-side comparisons. Click &ldquo;Benchmark all&rdquo; or use the benchmark button on each model card above.
                </p>
                <Button size="sm" variant="outline" className="h-8 text-xs" onClick={runAll} disabled={cmpLoading || cmpModels.length === 0}>
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
              <ComparisonTableCard completedResults={completedResults} models={cmpModels} bestMetrics={bestMetrics} />
              <ModelComparisonInsightsCard completedResults={completedResults} models={cmpModels} bestMetrics={bestMetrics} />
              <SummaryCard completedResults={completedResults} models={cmpModels} />
              <OutputComparisonCard models={cmpModels} />
              <VisualComparisonCard chartData={chartData} />
            </>
          )}
        </>
      )}
    </PageContainer>
  )
}
