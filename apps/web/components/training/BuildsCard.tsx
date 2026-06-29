'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui'
import { cn } from '@/lib/cn'
import { modelController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

export function BuildsCard({
  checkpoints,
  loadingTimedOut,
  onRetry,
}: {
  checkpoints: UseTrainingCheckpointsReturn
  loadingTimedOut: boolean
  onRetry: () => void
}) {
  const addToast = useToastStore(s => s.addToast)
  const hasBuilds = checkpoints.builds.length > 0 || checkpoints.loadingBuilds
  if (!hasBuilds) return null

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Builds</CardTitle>
        <Button size="sm" variant="ghost" onClick={() => checkpoints.fetchBuilds()}>
          Refresh
        </Button>
      </CardHeader>
      <CardContent className="p-0">
        {checkpoints.loadingBuilds ? (
          loadingTimedOut ? (
            <div className="px-4 py-6 text-center space-y-2">
              <p className="text-sm text-muted-foreground">Taking longer than expected</p>
              <Button size="sm" variant="ghost" onClick={onRetry}>
                Retry
              </Button>
            </div>
          ) : (
            <div className="divide-y divide-border/50">
              {[1,2].map(i => (
                <div key={i} className="flex items-center justify-between px-4 py-3">
                  <div className="space-y-1.5 flex-1">
                    <Skeleton className="h-4 w-56" />
                    <Skeleton className="h-3 w-40" />
                  </div>
                  <Skeleton className="h-5 w-14 rounded" />
                </div>
              ))}
            </div>
          )
        ) : (
          <div className="divide-y divide-border/50">
            {checkpoints.builds.slice().reverse().map((b, i) => (
              <div key={`${b.build_type}-${b.name}-${i}`} className="flex items-center justify-between px-4 py-3 text-sm">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="truncate font-medium">{b.name}</span>
                    <span className={cn(
                      'text-[10px] uppercase px-1.5 py-0.5 rounded font-medium',
                      b.build_type === 'auto-train' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300' : '',
                      b.build_type === 'hf-finetune' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300' : '',
                      b.build_type === 'hf-finetuned-dir' ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300' : '',
                      b.build_type === 'lora' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300' : '',
                      b.build_type === 'vlm' ? 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300' : '',
                    )}>
                      {b.build_type === 'auto-train' ? 'Auto' : b.build_type === 'hf-finetune' ? 'HF' : b.build_type === 'hf-finetuned-dir' ? 'Dir' : b.build_type === 'vlm' ? 'Visual' : 'Adv'}
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-muted-foreground mt-0.5">
                    {b.model && <span>Model: {b.model}</span>}
                    {b.dataset && <span>Dataset: {b.dataset}</span>}
                    {b.loss != null && <span>Loss: {Number(b.loss).toFixed(4)}</span>}
                    {b.epochs != null && <span>{b.epochs} epochs</span>}
                    {b.size_mb != null && <span>{(b.size_mb).toFixed(1)} MB</span>}
                    {b.training_dataset && b.training_dataset !== 'gpt2-generated' && <span>Data: {b.training_dataset}</span>}
                  </div>
                </div>
                <div className="flex items-center gap-1 shrink-0 ml-2">
                  {b.build_type === 'auto-train' && checkpoints.checkpoints.find(cp => cp.name === b.name) && (
                    checkpoints.checkpoints.find(cp => cp.name === b.name)!.name === checkpoints.activeCheckpoint
                      ? <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary">Active</span>
                      : <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => checkpoints.handleLoadCheckpoint(b.name, addToast)}>Load</Button>
                  )}
                  {b.build_type === 'vlm' && (
                    <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={async () => {
                      try { await modelController.loadVisualModel(b.model_path!); addToast(`Vision model loaded: ${b.name}`, 'success') }
                      catch { addToast('Failed to load vision model', 'error') }
                    }}>
                      Visual Chat
                    </Button>
                  )}
                  {b.model_path && b.build_type !== 'vlm' && (
                    <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={async () => {
                      try {
                        await modelController.loadModelPath(b.model_path!)
                        addToast(`Loaded ${b.name}`, 'success')
                      } catch { addToast('Failed to load model', 'error') }
                    }}>
                      Use
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
