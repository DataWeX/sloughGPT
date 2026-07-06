'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'

interface DPOCardProps {
  dpoRunning: boolean
  dpoStatus: string
  dpoResult: any
  dpoError: string | null
  dpoAccepted: number
  dpoRejected: number
  onTrigger: () => void
}

export default function DPOCard({ dpoRunning, dpoStatus, dpoResult, dpoError, dpoAccepted, dpoRejected, onTrigger }: DPOCardProps) {
  return (
    <Card>
      <CardHeader><CardTitle className="text-base">DPO fine-tune</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Run Direct Preference Optimization on feedback pairs (thumbs up/down) to align the model.
        </p>
        <div className="flex items-center gap-2">
          <Button size="sm" className="h-8 text-xs shrink-0" onClick={onTrigger} disabled={dpoRunning || dpoStatus === 'running'}>
            {dpoRunning || dpoStatus === 'running' ? 'Running…' : 'Run DPO'}
          </Button>
          {(dpoAccepted > 0 || dpoRejected > 0) && (
            <span className="text-xs text-muted-foreground">{dpoAccepted} accepted / {dpoRejected} rejected</span>
          )}
        </div>
        {dpoStatus === 'running' && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="animate-pulse h-2 w-2 rounded-full bg-primary" />
            DPO training in progress…
          </div>
        )}
        {dpoResult && dpoResult.status === 'accepted' && (
          <div className="space-y-1 p-2 rounded bg-success/10 border border-success/20 text-xs">
            <p className="text-success font-medium">✓ DPO accepted — model updated</p>
            {dpoResult.steps > 0 && <p className="text-muted-foreground">{dpoResult.steps} steps · avg loss {dpoResult.avg_loss?.toFixed(4)}</p>}
            {dpoResult.ppl_before != null && (
              <p className="text-muted-foreground">PPL: {dpoResult.ppl_before?.toFixed(2)} → {dpoResult.ppl_after?.toFixed(2)} ({dpoResult.ppl_delta_pct > 0 ? '+' : ''}{dpoResult.ppl_delta_pct?.toFixed(1)}%)</p>
            )}
            {dpoResult.pairs_trained > 0 && <p className="text-muted-foreground">{dpoResult.pairs_trained} pairs trained</p>}
            {dpoResult.elapsed_seconds > 0 && <p className="text-muted-foreground">Took {dpoResult.elapsed_seconds}s</p>}
          </div>
        )}
        {dpoResult && dpoResult.status === 'rejected' && (
          <div className="p-2 rounded bg-destructive/10 border border-destructive/20 text-xs text-destructive">
            DPO rejected — PPL degradation above threshold
          </div>
        )}
        {dpoError && (
          <div className="p-2 rounded bg-destructive/10 border border-destructive/20 text-xs text-destructive">{dpoError}</div>
        )}
      </CardContent>
    </Card>
  )
}
