'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Textarea, StatCard, KpiGrid, cn, Spinner } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { benchmarkController, type BenchmarkResult, type LoggedBenchmarkResponse } from '@/lib/benchmark-controller'
import { modelController } from '@/lib/model-controller'
import { apiPost } from '@/lib/http-client'
import { BenchmarkInsightsCard } from '@/components/benchmark/BenchmarkInsightsCard'
import { BenchmarkChartCard } from '@/components/benchmark/BenchmarkChartCard'
import { BenchmarkCompareCard } from '@/components/benchmark/BenchmarkCompareCard'
import { BenchmarkHistoryCard } from '@/components/benchmark/BenchmarkHistoryCard'
import ComparisonTableCard from '@/components/compare/ComparisonTableCard'
import { useToastStore } from '@/lib/toast-store'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'

type Tab = 'metrics' | 'quality' | 'responses' | 'perplexity' | 'compare'

export default function BenchmarkPage() {
  const router = useRouter()
  const [tab, setTab] = useState<Tab>('metrics')
  const [metrics, setMetrics] = useState<BenchmarkResult | null>(null)
  const [quality, setQuality] = useState<{ coherence_score: number; quality_score: number; repetition_rate: number; total_responses: number; avg_length: number } | null>(null)
  const [responses, setResponses] = useState<LoggedBenchmarkResponse[]>([])
  const [stats, setStats] = useState<{ total: number; avg_tokens: number } | null>(null)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [currentModel, setCurrentModel] = useState<string>('gpt2')

  const [pplxText, setPplxText] = useState('')
  const [pplxResult, setPplxResult] = useState<{ perplexity: number; loss: number; tokens: number } | null>(null)
  const [pplxLoading, setPplxLoading] = useState(false)
  const [compareModels, setCompareModels] = useState<string[]>([])
  const [compareResults, setCompareResults] = useState<[string, BenchmarkResult][]>([])
  const [compareLoading, setCompareLoading] = useState(false)
  const [availableModels, setAvailableModels] = useState<{ id: string; name: string }[]>([])
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => {
    const loadBenchmark = async () => {
      let model = 'gpt2'
      try {
        const h = await modelController.getHealth()
        model = h?.model_type ?? 'gpt2'
        setCurrentModel(model)
      } catch { /* use default */ }

      try {
        const models = await modelController.list()
        setAvailableModels(models.map(m => ({ id: m.id, name: m.name ?? m.id })))
      } catch { /* ignore */ }

      try {
        const [m, q, s] = await Promise.all([
          benchmarkController.metrics(model).catch(() => null),
          benchmarkController.quality().catch(() => null),
          benchmarkController.stats().catch(() => null),
        ])
        setMetrics(m)
        setQuality(q)
        setStats(s)
        if (!m && !q && !s) setLoadError('Could not load benchmark data. Please try again.')
      } catch {
        setLoadError('Could not load benchmark data. Please try again.')
      } finally {
        setLoading(false)
      }
    }
    loadBenchmark()
  }, [])

  const handleRefreshMetrics = async () => {
    setRunning(true)
    try {
      const m = await benchmarkController.run({ model: currentModel })
      setMetrics(m)
    } catch {
      addToast('Could not run benchmark', 'error')
    } finally {
      setRunning(false)
    }
  }

  useRefreshShortcut(handleRefreshMetrics)

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

  const handleRunCompare = async () => {
    if (compareModels.length === 0) return
    setCompareLoading(true)
    setCompareResults([])
    try {
      const results: [string, BenchmarkResult][] = []
      for (const model of compareModels) {
        try {
          const r = await benchmarkController.run({ model })
          results.push([model, r])
        } catch {
          addToast(`Benchmark failed for ${model}`, 'error')
        }
      }
      setCompareResults(results)
    } finally {
      setCompareLoading(false)
    }
  }

  const bestMetrics: Record<string, number> = {}
  for (const [model, result] of compareResults) {
    for (const key of ['throughput_tokens_per_sec', 'memory_mb', 'inference_time_ms'] as const) {
      const val = Number(result[key] ?? 0)
      if (!bestMetrics[key] || (key === 'memory_mb' || key === 'inference_time_ms' ? val < bestMetrics[key] : val > bestMetrics[key])) {
        bestMetrics[key] = val
      }
    }
  }

  return (
    <PageContainer
      title="Benchmark"
      subtitle="Model evaluation metrics"
      loading={loading}
      error={loadError}
      onRetry={() => window.location.reload()}
    >
      <div className="flex gap-1 border-b border-border/30 pb-0">
        {(['metrics', 'quality', 'responses', 'perplexity', 'compare'] as Tab[]).map(t => (
          <button
            key={t}
            type="button"
            onClick={() => {
              setTab(t)
              if (t === 'responses') handleLoadResponses()
            }}
            className={cn('px-3 py-1.5 text-[10px] font-medium rounded-t transition-colors', tab === t ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground hover:text-foreground')}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'metrics' && (
        <>
          <BenchmarkInsightsCard metrics={metrics} quality={quality} stats={stats} />
          <BenchmarkChartCard history={responses.map(r => ({
            timestamp: r.timestamp ?? '',
            model: r.model ?? '',
            throughput: r.tokens_generated && r.duration_ms ? (r.tokens_generated / (r.duration_ms / 1000)) : undefined,
            latency: r.duration_ms ?? undefined,
            tokens: r.tokens_generated ?? undefined,
          }))} />
          <KpiGrid>
            <StatCard label="Model" value={String(metrics?.model ?? '—')} />
            <StatCard label="Inferences" value={String(metrics?.inference_count ?? 0)} />
            <StatCard label="Tokens/s" value={String(metrics?.tokens_per_second ?? 0)} />
            <StatCard label="Memory" value={`${metrics?.memory_mb ?? 0} MB`} />
            <StatCard label="Total Tokens" value={String(metrics?.total_tokens ?? 0)} />
            <StatCard label="Loaded" value={metrics?.model_loaded ? 'Yes' : 'No'} />
          </KpiGrid>
          <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-[11px] font-medium">Model Metrics</CardTitle>
              <Button size="sm" variant="ghost" onClick={handleRefreshMetrics} disabled={running} aria-label="Refresh metrics">
              <Spinner className="h-6 w-6 text-[10px]" />
            </Button>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            {metrics ? (
               <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5">
                {[
                  { label: 'Model', value: String(metrics.model ?? '—') },
                  { label: 'Inferences', value: String(metrics.inference_count ?? 0) },
                  { label: 'Total Tokens', value: String(metrics.total_tokens ?? 0) },
                  { label: 'Tokens/s', value: String(metrics.tokens_per_second ?? 0) },
                  { label: 'Memory', value: `${metrics.memory_mb ?? 0} MB` },
                  { label: 'Loaded', value: metrics.model_loaded ? 'Yes' : 'No' },
                ].map(s => (
                   <div key={s.label} className="rounded-lg bg-muted/20 p-2.5 text-center hover:bg-muted/20">
                     <div className="text-[11px] text-muted-foreground">{s.label}</div>
                     <div className="text-[11px] font-mono font-medium tabular-nums">{s.value}</div>
                   </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6 text-[10px] text-muted-foreground/60">
                 No metrics available. Is a model loaded?
                <div className="mt-2">
                  <Button size="sm" variant="outline" className="h-7 text-[11px]" onClick={() => router.push('/models')}>
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

      {tab === 'quality' && (
        <Card>
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-[11px] font-medium">Quality Metrics</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            {quality ? (
               <div className="grid grid-cols-1 sm:grid-cols-3 gap-1.5">
                 {[
                  { label: 'Coherence', value: `${(quality.coherence_score * 100).toFixed(1)}%`, color: 'text-success' },
                  { label: 'Quality', value: `${(quality.quality_score * 100).toFixed(1)}%`, color: 'text-primary' },
                  { label: 'Repetition', value: `${(quality.repetition_rate * 100).toFixed(1)}%`, color: quality.repetition_rate > 0.3 ? 'text-destructive' : 'text-muted-foreground' },
                ].map(s => (
                   <div key={s.label} className="rounded-lg bg-muted/20 p-2.5 text-center hover:bg-muted/20">
                     <div className="text-[11px] text-muted-foreground">{s.label}</div>
                     <div className={cn('text-[11px] font-mono font-medium tabular-nums', s.color)}>{s.value}</div>
                   </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6 text-[10px] text-muted-foreground/60 space-y-2">
                 <div>No quality data yet. Chat with the model to generate responses.</div>
                 <Button size="sm" variant="outline" className="h-7 text-[11px]" onClick={() => router.push('/chat')}>
                   Open Chat
                 </Button>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {tab === 'responses' && (
        <>
          <BenchmarkHistoryCard
            history={responses.map(r => ({
              timestamp: r.timestamp ?? '',
              model: r.model ?? '',
              throughput: r.tokens_generated && r.duration_ms ? (r.tokens_generated / (r.duration_ms / 1000)) : undefined,
              latency: r.duration_ms ?? undefined,
              tokens: r.tokens_generated ?? undefined,
            }))}
            onClear={handleClearHistory}
          />
          <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-[11px] font-medium">Logged Responses ({responses.length})</CardTitle>
            <div className="flex gap-1">
              <Button size="sm" variant="ghost" onClick={handleLoadResponses} aria-label="Refresh responses" className="h-6 text-[10px]">
                <IconRefresh className="h-4 w-4" />
              </Button>
              <Button size="sm" variant="ghost" className="h-6 text-[10px] text-destructive" onClick={handleClearHistory}>
                Clear
              </Button>
            </div>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            {responses.length === 0 ? (
              <div className="text-center py-6 text-[10px] text-muted-foreground/60 space-y-2">
                 <div>No responses logged yet.</div>
                 <Button size="sm" variant="outline" className="h-7 text-[11px]" onClick={() => router.push('/chat')}>
                   Open Chat
                 </Button>
              </div>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {responses.map((r, i) => (
                   <div key={i} className="rounded-lg border border-border/40 p-2.5 hover:bg-muted/20 text-sm">
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
        </>
      )}

      {tab === 'perplexity' && (
        <Card>
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-[11px] font-medium">Perplexity Calculator</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5 space-y-3">
            <Textarea
              value={pplxText}
              onChange={e => setPplxText(e.target.value)}
              placeholder="Enter text to calculate perplexity..."
              rows={3}
            />
              <Button size="sm" onClick={handleCalcPerplexity} disabled={pplxLoading || !pplxText.trim()} className="h-7 text-[11px]">
              {pplxLoading ? 'Calculating...' : 'Calculate'}
            </Button>
            {pplxResult && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-1.5">
                   <div className="rounded-lg bg-muted/20 p-2.5 text-center hover:bg-muted/20">
                   <div className="text-[11px] text-muted-foreground">Perplexity</div>
                   <div className="text-[11px] font-mono font-medium tabular-nums">{pplxResult.perplexity}</div>
                 </div>
                 <div className="rounded-lg bg-muted/20 p-2.5 text-center hover:bg-muted/20">
                   <div className="text-[11px] text-muted-foreground">Loss</div>
                   <div className="text-[11px] font-mono font-medium tabular-nums">{pplxResult.loss}</div>
                 </div>
                 <div className="rounded-lg bg-muted/20 p-2.5 text-center hover:bg-muted/20">
                   <div className="text-[11px] text-muted-foreground">Tokens</div>
                   <div className="text-[11px] font-mono font-medium tabular-nums">{pplxResult.tokens}</div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {tab === 'compare' && (
        <Card>
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-[11px] font-medium">Compare Models</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5 space-y-4">
            <div className="flex flex-wrap gap-2">
              {availableModels.map(m => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setCompareModels(prev =>
                    prev.includes(m.id) ? prev.filter(x => x !== m.id) : [...prev, m.id]
                  )}
                  className={cn(
                    'px-3 py-1.5 text-xs rounded-md border transition-colors',
                    compareModels.includes(m.id)
                      ? 'bg-primary/10 text-primary border-primary'
                      : 'text-muted-foreground border-border hover:text-foreground',
                  )}
                >
                  {m.name}
                </button>
              ))}
              {availableModels.length === 0 && (
                <span className="text-xs text-muted-foreground">No models available</span>
              )}
            </div>
            <Button size="sm" onClick={handleRunCompare} disabled={compareLoading || compareModels.length === 0} className="h-7 text-[11px]">
              {compareLoading ? 'Running benchmarks...' : `Run on ${compareModels.length} model${compareModels.length !== 1 ? 's' : ''}`}
            </Button>
            <BenchmarkCompareCard results={compareResults} />
            {compareResults.length === 0 && !compareLoading && (
              <div className="text-center py-6 text-[10px] text-muted-foreground/60">
                 Select models above and click Run to compare them side by side.
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </PageContainer>
  )
}
