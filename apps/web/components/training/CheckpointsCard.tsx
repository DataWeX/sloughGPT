'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui'
import { cn } from '@/lib/cn'
import { useToastStore } from '@/lib/toast-store'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

export function CheckpointsCard({
  checkpoints,
  loadingTimedOut,
  onRetry,
  onContinue,
  onTest,
}: {
  checkpoints: UseTrainingCheckpointsReturn
  loadingTimedOut: boolean
  onRetry: () => void
  onContinue: (name: string) => void
  onTest: () => void
}) {
  const addToast = useToastStore(s => s.addToast)
  const hasCheckpoints = checkpoints.checkpoints.length > 0 || checkpoints.loadingCheckpoints
  if (!hasCheckpoints) return null

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Trained models</CardTitle>
        <Button size="sm" variant="ghost" onClick={onTest}>
          Test model
        </Button>
      </CardHeader>
      <CardContent>
        {checkpoints.loadingCheckpoints && checkpoints.checkpoints.length === 0 ? (
          loadingTimedOut ? (
            <div className="py-6 text-center space-y-2">
              <p className="text-sm text-muted-foreground">Taking longer than expected</p>
              <Button size="sm" variant="ghost" onClick={onRetry}>
                Retry
              </Button>
            </div>
          ) : (
            <div className="grid gap-2 sm:grid-cols-2">
              {[1,2].map(i => (
                <div key={i} className="flex items-center justify-between rounded-lg border border-border/50 p-3">
                  <div className="space-y-1.5 flex-1">
                    <Skeleton className="h-4 w-40" />
                    <Skeleton className="h-3 w-28" />
                  </div>
                  <Skeleton className="h-5 w-12 rounded" />
                </div>
              ))}
            </div>
          )
        ) : (
          <div className="grid gap-2 sm:grid-cols-2">
            {checkpoints.checkpoints.slice().reverse().map((cp: any) => (
              <div key={cp.name} className={cn("flex items-center justify-between rounded-lg border p-3 text-sm", checkpoints.activeCheckpoint === cp.name ? "border-primary/30 bg-primary/5" : "border-border/50")}>
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium text-xs">{cp.name}</p>
                  {cp.description ? (
                    <p className="text-[11px] text-muted-foreground mt-0.5">{cp.description}</p>
                  ) : (
                    <div className="flex flex-wrap gap-x-2 gap-y-0.5 text-[11px] text-muted-foreground mt-0.5">
                      {cp.loss != null && <span>Loss: {cp.loss.toFixed(4)}</span>}
                      {cp.epochs_trained != null && <span>{cp.epochs_trained} epochs</span>}
                      {cp.training_dataset && cp.training_dataset !== 'gpt2-generated' && <span>Dataset: {cp.training_dataset}</span>}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-1 shrink-0 ml-2">
                  {checkpoints.activeCheckpoint === cp.name ? (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary font-medium">Active</span>
                  ) : (
                    <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => checkpoints.handleLoadCheckpoint(cp.name, addToast)}>Load</Button>
                  )}
                  <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => onContinue(cp.name)}>Continue</Button>
                  <Button size="sm" variant="ghost" className="h-6 text-xs text-destructive hover:text-destructive" onClick={() => checkpoints.handleDeleteCheckpoint(cp.name, addToast)}>Del</Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
