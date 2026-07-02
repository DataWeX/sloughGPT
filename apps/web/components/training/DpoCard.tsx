'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useToastStore } from '@/lib/toast-store'
import { apiGet, apiPost } from '@/lib/http-client'

export function DpoCard() {
  const addToast = useToastStore(s => s.addToast)
  const [dpoRunning, setDpoRunning] = useState(false)
  const [dpoResult, setDpoResult] = useState<{
    status: string; avg_loss: number | null; ppl_before: number | null; ppl_after: number | null;
    ppl_delta_pct: number | null; pairs_trained: number; elapsed_seconds: number
  } | null>(null)
  const dpoPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => () => {
    if (dpoPollRef.current) { clearInterval(dpoPollRef.current) }
  }, [])

  const startDPO = useCallback(async () => {
    setDpoRunning(true)
    setDpoResult(null)
    try {
      const res = await apiPost<{ status: string }>('/multimodal/dpo', { max_pairs: 100 })
      const poll = async () => {
        try {
          const status = await apiGet<{ status: string; result?: any }>('/multimodal/dpo/status')
          if (status.status === 'completed' && status.result) {
            setDpoResult(status.result as typeof dpoResult)
            setDpoRunning(false)
            if (dpoPollRef.current) { clearInterval(dpoPollRef.current); dpoPollRef.current = null }
            addToast('DPO training complete!', 'success')
          } else if (status.status === 'failed') {
            setDpoRunning(false)
            addToast('DPO training failed', 'error')
            if (dpoPollRef.current) { clearInterval(dpoPollRef.current); dpoPollRef.current = null }
          }
        } catch { /* poll */ }
      }
      dpoPollRef.current = setInterval(poll, 3000)
    } catch {
      setDpoRunning(false)
      addToast('Failed to start DPO training', 'error')
    }
  }, [addToast])

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Direct Preference Optimization</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Refine a vision model by training on preference pairs (accepted vs rejected).
        </p>
        {dpoRunning && (
          <div className="space-y-2" role="status" aria-live="polite">
            <p className="text-sm text-muted-foreground">Running DPO training...</p>
            <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-primary animate-pulse rounded-full" style={{ width: '50%' }} />
            </div>
          </div>
        )}
        {dpoResult && (
          <div className="rounded-lg border border-success/20 bg-success/5 p-3 space-y-1">
            <p className="text-sm font-medium text-success">DPO training complete</p>
            {dpoResult.avg_loss != null && <p className="text-xs text-muted-foreground">Avg loss: {dpoResult.avg_loss.toFixed(4)}</p>}
            {dpoResult.ppl_before != null && <p className="text-xs text-muted-foreground">Perplexity before: {dpoResult.ppl_before.toFixed(2)}</p>}
            {dpoResult.ppl_after != null && <p className="text-xs text-muted-foreground">Perplexity after: {dpoResult.ppl_after.toFixed(2)}</p>}
            {dpoResult.ppl_delta_pct != null && (
              <p className={`text-xs font-medium ${dpoResult.ppl_delta_pct < 0 ? ' text-success' : ' text-destructive'}`}>
                {dpoResult.ppl_delta_pct < 0 ? '↓' : '↑'} {Math.abs(dpoResult.ppl_delta_pct).toFixed(1)}% perplexity
              </p>
            )}
            <div className="flex gap-2 pt-1">
              <Button size="sm" variant="outline" onClick={() => setDpoResult(null)}>Dismiss</Button>
            </div>
          </div>
        )}
        <Button size="sm" disabled={dpoRunning} onClick={startDPO}>
          {dpoRunning ? 'Training...' : 'Run DPO training'}
        </Button>
      </CardContent>
    </Card>
  )
}
