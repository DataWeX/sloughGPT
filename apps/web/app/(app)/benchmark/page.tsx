'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Textarea, StatCard, KpiGrid } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { benchmarkController, type BenchmarkResult, type LoggedBenchmarkResponse } from '@/lib/benchmark-controller'
import { modelController } from '@/lib/model-controller'
import { apiPost } from '@/lib/http-client'
import { BenchmarkInsightsCard } from '@/components/benchmark/BenchmarkInsightsCard'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

type Tab = 'metrics' | 'quality' | 'responses' | 'perplexity'

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
        const [m, q, s] = await Promise.all([
          benchmarkController.metrics(model).catch((e) => { logger.warning('benchmark metrics failed', { exception: String(e) }); return null }),
          benchmarkController.quality().catch((e) => { logger.warning('benchmark quality failed', { exception: String(e) }); return null }),
          benchmarkController.stats().catch((e) => { logger.warning('benchmark stats failed', { exception: String(e) }); return null }),
        ])
        setMetrics(m)
        setQuality(q)
        setStats(s)
        if (!m && !q && !s) setLoadError('Could not load benchmark data. Is the server running?')
      } catch {
        setLoadError('Could not load benchmark data. Is the server running?')
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
      addToast('Failed to run benchmark', 'error')
    } finally {
      setRunning(false)
    }
  }

  const handleLoadResponses = async () => {
    try {
      const data = await benchmarkController.history(20)
      setResponses(data)
    } catch {
      addToast('Failed to load responses', 'error')
    }
  }

  const handleClearHistory = async () => {
    try {
      await apiPost('/benchmark/history/clear', {})
      setResponses([])
      setStats(null)
    } catch {
      addToast('Failed to clear history', 'error')
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
      addToast('Failed to calculate perplexity', 'error')
    } finally {
      setPplxLoading(false)
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
        {(['metrics', 'quality', 'responses', 'perplexity'] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => {
              setTab(t)
              if (t === 'responses') handleLoadResponses()
            }}
            className={`px-3 py-1.5 text-xs font-medium rounded-t transition-colors ${
              tab === t ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'metrics' && (
        <>
          <BenchmarkInsightsCard metrics={metrics} quality={quality} stats={stats} />
          <KpiGrid>
            <StatCard label="Model" value={String(metrics?.model ?? '—')} />
            <StatCard label="Inferences" value={String(metrics?.inference_count ?? 0)} />
            <StatCard label="Tokens/s" value={String(metrics?.tokens_per_second ?? 0)} />
            <StatCard label="Memory" value={`${metrics?.memory_mb ?? 0} MB`} />
            <StatCard label="Total Tokens" value={String(metrics?.total_tokens ?? 0)} />
            <StatCard label="Loaded" value={metrics?.model_loaded ? 'Yes' : 'No'} />
          </KpiGrid>
          <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Model Metrics</CardTitle>
            <Button size="sm" variant="ghost" onClick={handleRefreshMetrics} disabled={running}>
              <IconRefresh className={`h-4 w-4 ${running ? 'animate-spin' : ''}`} />
            </Button>
          </CardHeader>
          <CardContent>
            {metrics ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {[
                  { label: 'Model', value: String(metrics.model ?? '—') },
                  { label: 'Inferences', value: String(metrics.inference_count ?? 0) },
                  { label: 'Total Tokens', value: String(metrics.total_tokens ?? 0) },
                  { label: 'Tokens/s', value: String(metrics.tokens_per_second ?? 0) },
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
                No metrics available. Is a model loaded?
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

      {tab === 'quality' && (
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
                    <div className={`text-lg font-mono font-medium ${s.color}`}>{s.value}</div>
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

      {tab === 'responses' && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Logged Responses ({responses.length})</CardTitle>
            <div className="flex gap-1">
              <Button size="sm" variant="ghost" onClick={handleLoadResponses}>
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

      {tab === 'perplexity' && (
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
    </PageContainer>
  )
}
