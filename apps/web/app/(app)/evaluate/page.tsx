'use client'
export const dynamic = 'force-dynamic'

import { useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Button, cn } from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Textarea } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { modelController } from '@/lib/model-controller'
import { benchmarkController, type BenchmarkResult, type LoggedBenchmarkResponse } from '@/lib/benchmark-controller'
import { apiPost } from '@/lib/http-client'
import { BenchmarkInsightsCard } from '@/components/benchmark/BenchmarkInsightsCard'
import { useToastStore } from '@/lib/toast-store'
import { useComparison } from '@/hooks/useComparison'
import { ComparisonView, ComparisonHeader } from '@/components/compare/ComparisonView'

type SingleModelTab = 'metrics' | 'quality' | 'responses' | 'perplexity'

export default function EvaluatePage() {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const [section, setSection] = useState<'single' | 'compare'>('single')

  const cmp = useComparison()

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

  // ── Shared header ──
  const headerRight = (
    <div className="flex items-center gap-2">
      {section === 'compare' && (
        <ComparisonHeader
          completedResults={cmp.completedResults}
          snapshotName={cmp.snapshotName}
          onSnapshotNameChange={cmp.setSnapshotName}
          onSaveSnapshot={cmp.saveSnapshot}
          onExport={cmp.exportResults}
          onRunAll={cmp.runAll}
          loading={cmp.loading}
          running={cmp.running}
        />
      )}
      <Button variant="outline" size="sm" onClick={section === 'single' ? handleRefreshMetrics : cmp.runAll} disabled={section === 'single' ? smRunning : cmp.loading || cmp.running.size > 0}>
        <IconRefresh className={cn('h-3.5 w-3.5 mr-1', (section === 'single' ? smRunning : cmp.running.size > 0) && 'animate-spin')} />
        {section === 'single' ? 'Refresh' : 'Benchmark all'}
      </Button>
    </div>
  )

  const loading = section === 'single' ? smLoading : cmp.loading
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
                        <div className={cn('text-base font-mono font-medium', s.color)}>{s.value}</div>
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
                      <div className="text-base font-mono font-medium">{pplxResult.perplexity}</div>
                    </div>
                    <div className="rounded-md bg-muted/30 p-3 text-center">
                      <div className="text-xs text-muted-foreground">Loss</div>
                      <div className="text-base font-mono font-medium">{pplxResult.loss}</div>
                    </div>
                    <div className="rounded-md bg-muted/30 p-3 text-center">
                      <div className="text-xs text-muted-foreground">Tokens</div>
                      <div className="text-base font-mono font-medium">{pplxResult.tokens}</div>
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
        <ComparisonView
          models={cmp.models}
          loading={cmp.loading}
          results={cmp.results}
          running={cmp.running}
          snapshots={cmp.snapshots}
          snapshotName={cmp.snapshotName}
          onSnapshotNameChange={cmp.setSnapshotName}
          completedResults={cmp.completedResults}
          bestMetrics={cmp.bestMetrics}
          chartData={cmp.chartData}
          onBenchmark={cmp.runBenchmark}
          onClear={cmp.clearResult}
          onRunAll={cmp.runAll}
          onExport={cmp.exportResults}
          onSaveSnapshot={cmp.saveSnapshot}
          onLoadSnapshot={cmp.loadSnapshot}
          onDeleteSnapshot={cmp.deleteSnapshot}
        />
      )}
    </PageContainer>
  )
}
