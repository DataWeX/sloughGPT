'use client'

import { useCallback, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@/components/ui/collapsible'
import { useToastStore } from '@/lib/toast-store'
import { userAdaptersController } from '@/lib/user-adapters-controller'
import type { UserAdapterStats } from '@/lib/user-adapters-controller'

export function AdapterQualityCard() {
  const addToast = useToastStore(s => s.addToast)
  const [adapterStats, setAdapterStats] = useState<UserAdapterStats | null>(null)
  const [aggregating, setAggregating] = useState(false)
  const [aggResult, setAggResult] = useState<Awaited<ReturnType<typeof userAdaptersController.aggregateBest>> | null>(null)

  const fetchAdapterStats = useCallback(async () => {
    try { setAdapterStats(await userAdaptersController.list()) } catch {}
  }, [])

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-base">Adapter Quality</CardTitle>
          <p className="text-xs text-muted-foreground mt-0.5">
            Feedback loop adapters — aggregate to measure quality delta
          </p>
        </div>
        <Button size="sm" disabled={aggregating} onClick={async () => {
          setAggregating(true)
          setAggResult(null)
          try {
            const res = await userAdaptersController.aggregateBest({ top_k: 10, min_feedback_count: 5 })
            setAggResult(res)
            addToast(res.eval?.verdict === 'accepted' ? 'Aggregation accepted — quality improved' : 'Aggregation complete', res.eval?.verdict === 'accepted' ? 'success' : 'info')
          } catch {
            addToast('Aggregation failed', 'error')
          } finally { setAggregating(false); void fetchAdapterStats() }
        }}>
          {aggregating ? 'Aggregating...' : 'Aggregate adapters'}
        </Button>
      </CardHeader>
      <CardContent className="space-y-3">
        {adapterStats ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="rounded-lg border border-border/50 p-3">
              <p className="text-xs text-muted-foreground">Total users</p>
              <p className="text-base font-semibold">{adapterStats.total_users}</p>
            </div>
            <div className="rounded-lg border border-border/50 p-3">
              <p className="text-xs text-muted-foreground">Total size</p>
              <p className="text-base font-semibold">{adapterStats.total_size_mb.toFixed(1)} MB</p>
            </div>
            <div className="rounded-lg border border-border/50 p-3">
              <p className="text-xs text-muted-foreground">Avg per user</p>
              <p className="text-base font-semibold">{adapterStats.avg_size_per_user_kb.toFixed(0)} KB</p>
            </div>
            <div className="rounded-lg border border-border/50 p-3">
              <p className="text-xs text-muted-foreground">Quality adapters</p>
              <p className="text-base font-semibold">{adapterStats.auto_management?.quality_adapters_count ?? 0}</p>
            </div>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">Loading adapter stats...</p>
        )}
        {aggResult && aggResult.eval && (
          <div className="rounded-lg border p-3 space-y-1">
            <div className="flex items-center gap-2">
              <span className={`text-sm font-medium ${aggResult.eval.verdict === 'accepted' ? 'text-success' : 'text-destructive'}`}>
                {aggResult.eval.verdict === 'accepted' ? '✓ Accepted' : '✗ Rejected'}
              </span>
              {aggResult.user_count != null && (
                <span className="text-xs text-muted-foreground">({aggResult.user_count} user adapters)</span>
              )}
            </div>
            {aggResult.eval.perplexity_delta != null && (
              <p className="text-xs text-muted-foreground">
                Perplexity Δ: <span className={aggResult.eval.perplexity_delta < 0 ? 'text-success' : 'text-destructive'}>
                  {aggResult.eval.perplexity_delta > 0 ? '+' : ''}{aggResult.eval.perplexity_delta.toFixed(2)}
                </span>
                {aggResult.eval.bleu_delta != null && (
                  <> &middot; BLEU Δ: <span className={aggResult.eval.bleu_delta > 0 ? 'text-success' : 'text-destructive'}>
                    {aggResult.eval.bleu_delta > 0 ? '+' : ''}{aggResult.eval.bleu_delta.toFixed(2)}
                  </span></>
                )}
              </p>
            )}
            {aggResult.eval.report && (
              <Collapsible className="text-xs">
                <CollapsibleTrigger className="cursor-pointer text-muted-foreground hover:text-foreground">View full report</CollapsibleTrigger>
                <CollapsibleContent>
                  <pre className="mt-1 whitespace-pre-wrap font-mono text-[11px] text-muted-foreground max-h-40 overflow-y-auto rounded bg-muted/50 p-2">{aggResult.eval.report}</pre>
                </CollapsibleContent>
              </Collapsible>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
