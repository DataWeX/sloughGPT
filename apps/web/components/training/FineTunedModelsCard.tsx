'use client'

import { useCallback, useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { cn, Card, CardContent, CardHeader, CardTitle, Button, Skeleton } from '@sloughgpt/strui'
import { IconTrash, IconRefresh, IconX } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { trainingJobsController, type FineTunedModel } from '@/lib/training-controller'
import { modelController } from '@/lib/model-controller'

export function FineTunedModelsCard({
  activeModelId,
  onLoaded,
}: {
  activeModelId?: string | null
  onLoaded?: () => void
}) {
  const addToast = useToastStore(s => s.addToast)
  const router = useRouter()
  const [models, setModels] = useState<FineTunedModel[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [loadingName, setLoadingName] = useState<string | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [batchDeleting, setBatchDeleting] = useState(false)

  const fetchModels = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const list = await trainingJobsController.listFineTuned()
      setModels(list)
    } catch (e) {
      setError(extractErrorMessage(e, 'Failed to load models'))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void fetchModels() }, [fetchModels])

  const handleLoad = async (name: string) => {
    setLoadingName(name)
    try {
      await trainingJobsController.loadFineTuned(name)
      addToast(`${name} loaded for chat`, 'success')
      onLoaded?.()
    } catch (e) {
      addToast(extractErrorMessage(e, 'Load failed'), 'error')
    } finally {
      setLoadingName(null)
    }
  }

  const handleDelete = async (name: string) => {
    try {
      await trainingJobsController.deleteFineTuned(name)
      addToast(`Deleted ${name}`, 'success')
      void fetchModels()
    } catch (e) {
      addToast(extractErrorMessage(e, 'Delete failed'), 'error')
    }
  }

  const handleUnload = async (name: string) => {
    setLoadingName(name)
    try {
      await modelController.unloadModel(name)
      addToast(`${name} unloaded`, 'info')
      onLoaded?.()
    } catch (e) {
      addToast(extractErrorMessage(e, 'Unload failed'), 'error')
    } finally {
      setLoadingName(null)
    }
  }

  const toggleSelect = (name: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  const toggleSelectAll = () => {
    if (selectedIds.size === models.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(models.map(m => m.name)))
    }
  }

  const handleBatchDelete = async () => {
    if (selectedIds.size === 0) return
    setBatchDeleting(true)
    try {
      await Promise.all(Array.from(selectedIds).map(name => trainingJobsController.deleteFineTuned(name)))
      setSelectedIds(new Set())
      addToast(`Deleted ${selectedIds.size} models`, 'success')
      void fetchModels()
    } catch {
      addToast('Batch delete failed', 'error')
    } finally {
      setBatchDeleting(false)
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Fine-tuned models</CardTitle>
        <Button size="sm" variant="ghost" onClick={() => void fetchModels()} aria-label="Refresh fine-tuned models">
          <IconRefresh className="h-3.5 w-3.5" />
        </Button>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="grid gap-2 sm:grid-cols-2">
            {[1, 2].map(i => (
              <div key={i} className="flex items-center justify-between rounded-lg border border-border/50 p-3">
                <div className="space-y-1.5 flex-1">
                  <Skeleton className="h-4 w-40" />
                  <Skeleton className="h-3 w-28" />
                </div>
                <Skeleton className="h-5 w-12 rounded" />
              </div>
            ))}
          </div>
        ) : error ? (
          <div className="text-center py-4">
            <p className="text-xs text-destructive mb-2">{error}</p>
            <Button size="sm" variant="ghost" onClick={() => void fetchModels()}>Retry</Button>
          </div>
        ) : models.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            No fine-tuned models yet. HF fine-tuned outputs under models/hf-finetuned appear here.
          </p>
        ) : (
          <>
            {selectedIds.size > 0 && (
              <div className="flex items-center gap-2 rounded-md bg-destructive/5 border border-destructive/20 px-3 py-2 mb-2">
                <span className="text-xs text-destructive font-medium">{selectedIds.size} selected</span>
                <Button size="sm" variant="ghost" className="text-destructive h-6 text-[10px] ml-auto" onClick={handleBatchDelete} disabled={batchDeleting}>
                  {batchDeleting ? 'Deleting...' : 'Delete Selected'}
                </Button>
                <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => setSelectedIds(new Set())}>
                  Clear
                </Button>
              </div>
            )}
            {models.length > 2 && (
              <label className="flex items-center gap-2 text-[10px] text-muted-foreground cursor-pointer mb-2">
                <input
                  type="checkbox"
                  checked={selectedIds.size === models.length && models.length > 0}
                  onChange={toggleSelectAll}
                  className="rounded border-border"
                />
                Select all ({models.length})
              </label>
            )}
            <div className="grid gap-2 sm:grid-cols-2">
              {models.map((m) => {
                const isActive = !!activeModelId && (activeModelId === m.model_name || activeModelId === m.name)
                const isSelected = selectedIds.has(m.name)
                return (
                  <div key={m.name} className={cn(
                    'flex items-center justify-between rounded-lg border p-3 text-sm transition-colors',
                    isActive ? 'border-primary/30 bg-primary/[0.08]' : isSelected ? 'border-primary/40 bg-primary/5' : 'border-border/50 hover:bg-muted/30',
                  )}>
                    <div className="flex items-start gap-2 min-w-0 flex-1">
                      {models.length > 2 && (
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleSelect(m.name)}
                          className="mt-1 rounded border-border shrink-0"
                        />
                      )}
                      <div className="min-w-0 flex-1">
                        <button
                          onClick={() => router.push(`/model/${encodeURIComponent(m.name)}`)}
                          className="flex items-center gap-2 text-left w-full"
                          aria-label={`View details for ${m.name}`}
                        >
                          <p className="truncate font-medium text-xs hover:text-primary transition-colors">{m.name}</p>
                          {isActive && <span className="text-primary text-[10px]">✓</span>}
                        </button>
                        <div className="flex flex-wrap gap-x-2 gap-y-0.5 text-[11px] text-muted-foreground mt-0.5">
                          <span>{m.model}</span>
                          {m.dataset && <span>· {m.dataset}</span>}
                          {m.size_mb > 0 && <span>· {m.size_mb.toFixed(1)} MB</span>}
                          {m.final_loss != null && <span>· loss {Number(m.final_loss).toFixed(4)}</span>}
                          {m.epochs && m.epochs > 0 && <span>· {m.epochs} ep</span>}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-1 shrink-0 ml-2">
                      {!isActive && (
                        <Button size="sm" variant="ghost" className="h-6 text-xs" disabled={loadingName !== null} onClick={() => handleLoad(m.name)}>
                          {loadingName === m.name ? 'Loading...' : 'Load'}
                        </Button>
                      )}
                      {isActive && (
                        <Button size="sm" variant="ghost" className="h-6 text-xs" disabled={loadingName !== null} onClick={() => handleUnload(m.name)} aria-label={`Unload ${m.name}`}>
                          {loadingName === m.name ? 'Unloading...' : (
                            <>
                              <IconX className="h-3 w-3 mr-1" />
                              Unload
                            </>
                          )}
                        </Button>
                      )}
                      <Button size="sm" variant="ghost" className="h-6 text-xs text-destructive hover:text-destructive" onClick={() => handleDelete(m.name)} aria-label={`Delete ${m.name}`}>
                        <IconTrash className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                )
              })}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
