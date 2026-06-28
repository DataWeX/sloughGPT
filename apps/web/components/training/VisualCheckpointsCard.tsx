'use client'

import { useCallback, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useToastStore } from '@/lib/toast-store'
import { visualController } from '@/lib/visual-controller'

export function VisualCheckpointsCard() {
  const addToast = useToastStore(s => s.addToast)
  const [visualCheckpoints, setVisualCheckpoints] = useState<any[]>([])
  const [loadingVisualCkpts, setLoadingVisualCkpts] = useState(false)
  const [loadingVisualCkptName, setLoadingVisualCkptName] = useState<string | null>(null)

  const fetchVisualCheckpoints = useCallback(async () => {
    setLoadingVisualCkpts(true)
    try { setVisualCheckpoints(await visualController.listCheckpoints()) } catch {}
    finally { setLoadingVisualCkpts(false) }
  }, [])

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Visual Checkpoints</CardTitle>
        <Button size="sm" variant="ghost" onClick={fetchVisualCheckpoints} disabled={loadingVisualCkpts}>
          {loadingVisualCkpts ? 'Loading...' : 'Refresh'}
        </Button>
      </CardHeader>
      <CardContent className="space-y-2">
        {visualCheckpoints.length === 0 ? (
          <p className="text-sm text-muted-foreground py-2">No visual checkpoints yet</p>
        ) : (
          <div className="grid gap-2 sm:grid-cols-2">
            {visualCheckpoints.map((ckpt: any) => {
              const isBusy = loadingVisualCkptName === ckpt.name
              return (
                <div key={ckpt.name} className="flex items-center justify-between rounded-lg border border-border/50 p-3">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium truncate">{ckpt.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {ckpt.model ? ckpt.model : 'vision'}
                      {ckpt.loss != null ? ` · ${ckpt.loss.toFixed(4)} loss` : ''}
                      {ckpt.epochs ? ` · ${ckpt.epochs} epochs` : ''}
                    </p>
                  </div>
                  <div className="flex items-center gap-1 ml-2 shrink-0">
                    <Button size="sm" variant="ghost" className="h-7 text-xs" disabled={isBusy} onClick={async () => {
                      setLoadingVisualCkptName(ckpt.name)
                      try {
                        await visualController.loadCheckpoint(ckpt.name)
                        addToast(`Loaded visual checkpoint: ${ckpt.name}`, 'success')
                      } catch { addToast('Failed to load visual checkpoint', 'error') }
                      finally { setLoadingVisualCkptName(null); void fetchVisualCheckpoints() }
                    }}>
                      {isBusy ? 'Loading...' : 'Load'}
                    </Button>
                    <Button size="sm" variant="ghost" className="h-7 text-xs text-destructive" disabled={isBusy} onClick={async () => {
                      if (!confirm(`Delete visual checkpoint "${ckpt.name}"?`)) return
                      setLoadingVisualCkptName(ckpt.name)
                      try {
                        await visualController.deleteCheckpoint(ckpt.name)
                        addToast(`Deleted: ${ckpt.name}`, 'info')
                      } catch { addToast('Failed to delete visual checkpoint', 'error') }
                      finally { setLoadingVisualCkptName(null); void fetchVisualCheckpoints() }
                    }}>
                      Delete
                    </Button>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
