'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { IconRefresh, IconTrash } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { userAdaptersController, type UserAdapterInfo, type UserAdapterStats } from '@/lib/user-adapters-controller'
import { extractErrorMessage } from '@/lib/error-utils'
import { useToastStore } from '@/lib/toast-store'
import { PUBLIC_API_URL } from '@/lib/config'

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
  const [evalHistory, setEvalHistory] = useState<Record<string, unknown>[]>([])
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
        const evalRes = await fetch(`${PUBLIC_API_URL}/lora-eval/history?limit=10`)
        const evalData = await evalRes.json()
        setEvalHistory(evalData?.data?.results ?? [])
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
    try {
      await userAdaptersController.reset(userId)
      await fetchData()
    } catch {
      addToast('Failed to reset adapter', 'error')
    }
  }

  const handleRunEval = async () => {
    setRunningEval(true)
    try {
      await fetch(`${PUBLIC_API_URL}/lora-eval/run?adapter_path=data/user_adapters/best_aggregated.npz`, { method: 'GET' })
      const evalRes = await fetch(`${PUBLIC_API_URL}/lora-eval/history?limit=10`)
      const evalData = await evalRes.json()
      setEvalHistory(evalData?.data?.results ?? [])
      addToast('Evaluation complete', 'success')
    } catch {
      addToast('Eval failed', 'error')
    } finally {
      setRunningEval(false)
    }
  }

  if (loading) {
    return (
      <div className="sl-page mx-auto max-w-4xl">
        <AppRouteHeader left={<AppRouteHeaderLead title="Adapters" subtitle="Per-user LoRA adapter management" />} />
        <div className="space-y-4">
          <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="sl-page mx-auto max-w-4xl">
        <AppRouteHeader left={<AppRouteHeaderLead title="Adapters" subtitle="Per-user LoRA adapter management" />} />
        <div className="space-y-4">
          <Card>
            <CardContent className="text-center py-8">
              <p className="text-sm text-destructive mb-2">{error}</p>
              <Button size="sm" variant="ghost" onClick={() => void fetchData()}>Retry</Button>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Adapters" subtitle="Per-user LoRA adapter management" />} />
      <div className="space-y-4">
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

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Actions</CardTitle>
            <Button size="sm" variant="ghost" onClick={fetchData}>
              <IconRefresh className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
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
              <p className="text-sm text-muted-foreground">
                No adapters yet. Use the chat to generate feedback that creates adapters.
              </p>
            ) : (
              <div className="space-y-2">
                {adapters.map(a => (
                  <div
                    key={a.user_id}
                    className="flex items-center justify-between rounded-md border border-border/60 px-3 py-2 text-sm hover:bg-muted/50 transition-colors"
                  >
                    <div className="min-w-0">
                      <div className="font-medium truncate">{a.user_id}</div>
                      <div className="text-xs text-muted-foreground">
                        {a.feedback_count} feedback · rank {a.rank}
                        {a.created_at && <> · {new Date(a.created_at).toLocaleDateString()}</>}
                      </div>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-destructive"
                      onClick={() => handleReset(a.user_id)}
                    >
                      <IconTrash className="h-3.5 w-3.5" />
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
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {evalHistory.map((r, i) => (
                  <div key={i} className="flex items-center gap-3 rounded-md bg-muted/20 p-2 text-xs">
                    <span className="font-medium">{String(r.adapter_path ?? '—')}</span>
                    <span className="text-muted-foreground">{String(r.status ?? '—')}</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
