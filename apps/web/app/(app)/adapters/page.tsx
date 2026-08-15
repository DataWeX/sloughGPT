'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { IconRefresh, IconTrash } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { userAdaptersController, type UserAdapterInfo, type UserAdapterStats } from '@/lib/user-adapters-controller'
import { loraEvalController, type LoraEvalResult } from '@/lib/lora-eval-controller'
import { AdapterHealthCard } from '@/components/adapters/AdapterHealthCard'
import { extractErrorMessage } from '@/lib/error-utils'
import { useToastStore } from '@/lib/toast-store'

export default function AdaptersPage() {
  const [stats, setStats] = useState<UserAdapterStats | null>(null)
  const [adapters, setAdapters] = useState<UserAdapterInfo[]>([])
  const [quality, setQuality] = useState<{ count: number; adapters: UserAdapterInfo[] } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const addToast = useToastStore(s => s.addToast)
  const [aggregating, setAggregating] = useState(false)
  const [aggregateResult, setAggregateResult] = useState<string | null>(null)
  const [pruning, setPruning] = useState(false)
  const [evalHistory, setEvalHistory] = useState<LoraEvalResult[]>([])
  const [runningEval, setRunningEval] = useState(false)

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [statsRes, qualityRes] = await Promise.all([
        userAdaptersController.list(),
        userAdaptersController.getQuality(3),
      ])
      setStats(statsRes)
      setQuality(qualityRes)
      setAdapters(qualityRes.adapters ?? [])
      try {
        const evalResults = await loraEvalController.getHistory(10)
        setEvalHistory(evalResults)
      } catch { /* optional */ }
    } catch (e) {
      setError(extractErrorMessage(e, 'Failed to load adapters'))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let ignore = false
    const load = async () => {
      setLoading(true)
      setError(null)
      try {
        const [statsRes, qualityRes] = await Promise.all([
          userAdaptersController.list(),
          userAdaptersController.getQuality(3),
        ])
        if (ignore) return
        setStats(statsRes)
        setQuality(qualityRes)
        setAdapters(qualityRes.adapters ?? [])
      } catch (e) {
        if (!ignore) setError(extractErrorMessage(e, 'Failed to load adapters'))
      } finally {
        if (!ignore) setLoading(false)
      }
    }
    load()
    return () => { ignore = true }
  }, [])

  const handleAggregate = async () => {
    setAggregating(true)
    setAggregateResult(null)
    try {
      const res = await userAdaptersController.aggregateBest()
      const evalInfo = res.eval
      if (evalInfo?.verdict) {
        setAggregateResult(`Aggregated ${res.user_count ?? 0} adapters. Verdict: ${evalInfo.verdict}`)
      } else {
        setAggregateResult(`Aggregated ${res.user_count ?? 0} adapters`)
      }
    } catch (err) {
      setAggregateResult(err instanceof Error ? err.message : 'Aggregation failed')
    } finally {
      setAggregating(false)
    }
  }

  const handlePrune = async () => {
    setPruning(true)
    try {
      const res = await userAdaptersController.prune()
      setAggregateResult(`Pruned ${res.deleted_count} adapters`)
      await fetchData()
    } catch (err) {
      setAggregateResult(err instanceof Error ? err.message : 'Prune failed')
    } finally {
      setPruning(false)
    }
  }

  const handleReset = async (userId: string) => {
    if (!window.confirm(`Reset adapter for ${userId}? This cannot be undone.`)) return
    try {
      await userAdaptersController.reset(userId)
      await fetchData()
      addToast(`Adapter for ${userId} reset`, 'success')
    } catch {
      addToast('Failed to reset adapter', 'error')
    }
  }

  const handleRunEval = async () => {
    setRunningEval(true)
    try {
      await loraEvalController.runEval('data/user_adapters/best_aggregated.npz')
      const evalResults = await loraEvalController.getHistory(10)
      setEvalHistory(evalResults)
      addToast('Evaluation complete', 'success')
    } catch {
      addToast('Eval failed', 'error')
    } finally {
      setRunningEval(false)
    }
  }

  if (loading) {
    return (
      <PageContainer title="Adapters" subtitle="Per-user LoRA adapter management" loadingCards={1}>
        <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
      </PageContainer>
    )
  }

  if (error) {
    return (
      <PageContainer title="Adapters" subtitle="Per-user LoRA adapter management" error={error} onRetry={() => void fetchData()}>
        <></>
      </PageContainer>
    )
  }

  return (
    <PageContainer
      title="Adapters"
      subtitle="Per-user LoRA adapter management"
      headerRight={<Button size="sm" variant="ghost" onClick={fetchData}><IconRefresh className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} /></Button>}
    >
      {aggregateResult && (
        <div className="rounded-md bg-primary/10 border border-primary/20 px-4 py-3 text-sm text-primary">
          {aggregateResult}
          <button className="ml-2 underline" onClick={() => setAggregateResult(null)}>Dismiss</button>
        </div>
      )}

      {stats && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Adapter Stats</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: 'Users', value: stats.total_users },
                { label: 'Total Size', value: `${stats.total_size_mb?.toFixed(1) ?? 0} MB` },
                { label: 'Rank', value: stats.adapter_rank },
                { label: 'Avg/User', value: `${stats.avg_size_per_user_kb?.toFixed(1) ?? 0} KB` },
              ].map(s => (
                <div key={s.label} className="rounded-md bg-muted/30 p-3 text-center">
                  <div className="text-xs text-muted-foreground">{s.label}</div>
                  <div className="text-lg font-mono font-medium">{s.value}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <AdapterHealthCard adapters={adapters} />

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-base">Actions</CardTitle>
          <Button size="sm" variant="ghost" onClick={fetchData}>
            <IconRefresh className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Button size="sm" onClick={handleAggregate} disabled={aggregating}>
              {aggregating ? 'Aggregating...' : 'Aggregate Best'}
            </Button>
            <Button size="sm" variant="outline" onClick={handlePrune} disabled={pruning}>
              {pruning ? 'Pruning...' : 'Prune Old'}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Adapters ({adapters.length})</CardTitle>
        </CardHeader>
        <CardContent>
          {adapters.length === 0 ? (
            <div className="text-center py-6 space-y-2">
              <p className="text-sm text-muted-foreground">No adapters yet.</p>
              <a href="/chat" className="text-sm text-primary hover:underline">Start a chat</a>
              <span className="text-sm text-muted-foreground"> and give feedback to create adapters.</span>
            </div>
          ) : (
            <div className="space-y-2">
              {adapters.map(a => (
                <div
                  key={a.user_id}
                  className="flex items-center justify-between rounded-md border border-border/60 px-3 py-2 text-sm hover:bg-muted/50 transition-colors"
                >
                  <div className="min-w-0 flex-1">
                    <div className="font-medium truncate">{a.user_id}</div>
                    <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-muted-foreground mt-0.5">
                      <span>{a.feedback_count} feedback</span>
                      <span>rank {a.rank}</span>
                      <span>alpha {a.alpha}</span>
                      <span>dim {a.model_dim}</span>
                      {a.updated_at && <span>updated {new Date(a.updated_at).toLocaleDateString()}</span>}
                    </div>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-destructive"
                    onClick={() => handleReset(a.user_id)}
                  >
                    <IconTrash className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Evaluation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <Button size="sm" onClick={handleRunEval} disabled={runningEval}>
            {runningEval ? 'Running...' : 'Run LoRA Eval'}
          </Button>
          {evalHistory.length > 0 && (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {evalHistory.map((r, i) => (
                <div key={i} className="rounded-md border border-border/60 p-3 text-xs space-y-1.5">
                  <div className="flex items-center gap-2">
                    <span className="font-medium truncate">{String(r.adapter_path ?? '—')}</span>
                    <span className={`px-1.5 py-0.5 rounded font-medium ${
                      r.verdict === 'accept' ? 'bg-success/15 text-success' :
                      r.verdict === 'reject' ? 'bg-destructive/15 text-destructive' :
                      'bg-muted text-muted-foreground'
                    }`}>
                      {String(r.verdict ?? '—')}
                    </span>
                    {r.timestamp && (
                      <span className="text-muted-foreground ml-auto">{new Date(r.timestamp).toLocaleString()}</span>
                    )}
                  </div>
                  {(r.perplexity != null || r.bleu != null) && (
                    <div className="flex gap-4 text-muted-foreground">
                      {r.perplexity != null && <span>Perplexity: {Number(r.perplexity).toFixed(3)}</span>}
                      {r.bleu != null && <span>BLEU: {Number(r.bleu).toFixed(3)}</span>}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </PageContainer>
  )
}
