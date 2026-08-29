'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback, useEffect } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, KpiGrid, StatCard } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { loraEvalController, type LoraEvalResult } from '@/lib/lora-eval-controller'

export default function LoraEvalPage() {
  const addToast = useToastStore(s => s.addToast)
  const [history, setHistory] = useState<LoraEvalResult[]>([])
  const [loading, setLoading] = useState(true)
  const [evalRunning, setEvalRunning] = useState(false)
  const [aggregating, setAggregating] = useState(false)
  const [lastResult, setLastResult] = useState<LoraEvalResult | null>(null)
  const [adapterPath, setAdapterPath] = useState('data/user_adapters/best_aggregated.npz')
  const [soul, setSoul] = useState('assistant')
  const [topK, setTopK] = useState(10)
  const [minFeedback, setMinFeedback] = useState(5)

  const fetchHistory = useCallback(async () => {
    setLoading(true)
    try {
      const resp = await loraEvalController.getHistory(50)
      setHistory(resp ?? [])
    } catch {
      addToast('Could not load eval history', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { void fetchHistory() }, [fetchHistory])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void fetchHistory() }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [fetchHistory])

  const handleRunEval = useCallback(async () => {
    setEvalRunning(true)
    try {
      const result = await loraEvalController.runEval(adapterPath, soul)
      setLastResult(result)
      addToast(`Eval complete: ${result.status}`, 'success')
      void fetchHistory()
    } catch {
      addToast('Could not run eval', 'error')
    } finally {
      setEvalRunning(false)
    }
  }, [adapterPath, soul, addToast, fetchHistory])

  const handleAggregate = useCallback(async () => {
    setAggregating(true)
    try {
      const result = await loraEvalController.aggregate(topK, minFeedback)
      addToast(`Aggregated: ${result.status}`, 'success')
      void fetchHistory()
    } catch {
      addToast('Could not aggregate adapters', 'error')
    } finally {
      setAggregating(false)
    }
  }, [topK, minFeedback, addToast, fetchHistory])

  return (
    <PageContainer
      title="LoRA Evaluation"
      subtitle="Evaluate adapter quality — baseline vs with-adapter comparison"
      headerRight={
        <Button size="sm" variant="ghost" onClick={() => void fetchHistory()}>Refresh</Button>
      }
    >
      <KpiGrid columns={3}>
        <StatCard label="Total Evals" value={String(history.length)} />
        <StatCard label="Last Status" value={lastResult?.status ?? history[0]?.status ?? '—'} />
        <StatCard label="Best Verdict" value={history.find(h => h.delta?.verdict)?.status ?? '—'} />
      </KpiGrid>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Run Evaluation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-muted-foreground">Adapter Path</label>
              <Input value={adapterPath} onChange={e => setAdapterPath(e.target.value)} className="h-8 text-xs font-mono mt-1" />
            </div>
            <div>
              <label className="text-xs text-muted-foreground">Soul</label>
              <Input value={soul} onChange={e => setSoul(e.target.value)} className="h-8 text-xs mt-1" />
            </div>
          </div>
          <Button size="sm" onClick={() => void handleRunEval()} disabled={evalRunning}>
            {evalRunning ? 'Running...' : 'Run Eval'}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Aggregate Adapters</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-muted-foreground">Top K</label>
              <Input type="number" value={topK} onChange={e => setTopK(parseInt(e.target.value) || 10)} className="h-8 text-xs mt-1" />
            </div>
            <div>
              <label className="text-xs text-muted-foreground">Min Feedback</label>
              <Input type="number" value={minFeedback} onChange={e => setMinFeedback(parseInt(e.target.value) || 5)} className="h-8 text-xs mt-1" />
            </div>
          </div>
          <Button size="sm" onClick={() => void handleAggregate()} disabled={aggregating}>
            {aggregating ? 'Aggregating...' : 'Aggregate'}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Eval History ({history.length})</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <p className="text-xs text-muted-foreground">Loading...</p>
          ) : history.length === 0 ? (
            <p className="text-xs text-muted-foreground">No evaluations yet. Run an eval above.</p>
          ) : (
            <div className="space-y-2 max-h-96 overflow-auto">
              {history.map((h, i) => (
                <div key={i} className="rounded bg-muted/30 px-3 py-2 text-xs space-y-1">
                  <div className="flex justify-between">
                    <span className="font-medium">{h.status}</span>
                    {h.elapsed_ms != null && <span className="text-muted-foreground">{h.elapsed_ms}ms</span>}
                  </div>
                  {h.report && <p className="text-muted-foreground whitespace-pre-wrap">{h.report}</p>}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </PageContainer>
  )
}
