'use client'

import { useMemo, useState } from 'react'
import { extractErrorMessage } from '@/lib/error-utils'
import { cn, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { IconSearch, IconTrash, IconDownload } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { downloadBlob, downloadJson } from '@/lib/download-utils'
import { trainingJobsController } from '@/lib/controllers'
import { todayDateString } from '@/lib/format-bytes'
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
  const [search, setSearch] = useState('')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [sortBy, setSortBy] = useState<'date' | 'loss' | 'name'>('date')
  const [loadingCheckpoint, setLoadingCheckpoint] = useState<string | null>(null)

  const filteredCheckpoints = useMemo(() => {
    const list = checkpoints.checkpoints.filter(cp =>
      !search || cp.name.toLowerCase().includes(search.toLowerCase()) ||
      (cp.description && cp.description.toLowerCase().includes(search.toLowerCase())) ||
      (cp.training_dataset && cp.training_dataset.toLowerCase().includes(search.toLowerCase()))
    )
    const sorted = [...list]
    if (sortBy === 'loss') sorted.sort((a, b) => (a.loss ?? Infinity) - (b.loss ?? Infinity))
    else if (sortBy === 'name') sorted.sort((a, b) => a.name.localeCompare(b.name))
    else sorted.reverse() // date = newest first (API returns chronological)
    return sorted
  }, [checkpoints.checkpoints, search, sortBy])

  const toggleSelect = (name: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  const selectAll = () => {
    if (selectedIds.size === filteredCheckpoints.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(filteredCheckpoints.map(cp => cp.name)))
    }
  }

  const bulkDelete = async () => {
    if (selectedIds.size === 0) return
    const names = Array.from(selectedIds)
    try {
      const result = await trainingJobsController.deleteCheckpointsBatch(names)
      setSelectedIds(new Set())
      addToast(`Deleted ${result.deleted} checkpoints`, 'success')
      checkpoints.fetchCheckpoints()
    } catch (e) {
      addToast(extractErrorMessage(e, 'Batch delete failed'), 'error')
    }
  }

  const hasCheckpoints = checkpoints.checkpoints.length > 0 || checkpoints.loadingCheckpoints
  if (checkpoints.checkpoints.length === 0 && !checkpoints.loadingCheckpoints) return (
    <Card>
      <CardHeader><CardTitle className="text-base">Trained models</CardTitle></CardHeader>
      <CardContent>
        <p className="text-xs text-muted-foreground text-center py-4">No trained models yet. Start training to create your first checkpoint.</p>
      </CardContent>
    </Card>
  )

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
          <>
            {checkpoints.checkpoints.length > 2 && (
              <div className="flex items-center gap-2 mb-3">
                <div className="relative flex-1 max-w-xs">
                  <IconSearch className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                  <input
                    value={search}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearch(e.target.value)}
                    placeholder="Search checkpoints..."
                    className="h-7 w-full rounded-md border border-border bg-background pl-7 pr-2 text-xs"
                  />
                </div>
                <div className="flex gap-1">
                  {(['date', 'loss', 'name'] as const).map(s => (
                    <button
                      key={s}
                      onClick={() => setSortBy(s)}
                      className={`text-[10px] px-2 py-0.5 rounded border transition-colors ${sortBy === s ? 'bg-primary/15 text-primary border-primary/30' : 'border-border/40 text-muted-foreground hover:bg-muted/80'}`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
                {selectedIds.size > 0 && (
                  <>
                    <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => {
                      const selected = checkpoints.checkpoints.filter(cp => selectedIds.has(cp.name))
                      const data = selected.map(cp => ({
                        name: cp.name,
                        loss: cp.loss,
                        epochs: cp.epochs_trained,
                        dataset: cp.training_dataset,
                        duration_s: cp.training_duration_s,
                        size_mb: cp.size_mb,
                        model_type: cp.model_type,
                      }))
                      downloadJson(data, `checkpoints-export-${todayDateString()}.json`)
                      addToast(`Exported ${selected.length} checkpoint metadata`, 'success')
                    }}>
                      Export {selectedIds.size}
                    </Button>
                    <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={async () => {
                      for (const name of selectedIds) {
                        try {
                          const blob = await trainingJobsController.downloadCheckpoint(name)
                          downloadBlob(blob, `${name}.sou`)
                        } catch { /* skip */ }
                      }
                      addToast(`Downloaded ${selectedIds.size} checkpoints`, 'success')
                    }}>
                      <IconDownload className="h-3 w-3 mr-1" />
                      Download {selectedIds.size}
                    </Button>
                    <Button size="sm" variant="destructive" className="h-6 text-[10px]" onClick={bulkDelete}>
                      <IconTrash className="h-3 w-3 mr-1" />
                      Delete {selectedIds.size}
                    </Button>
                  </>
                )}
                <button type="button" onClick={selectAll} className="text-[10px] text-muted-foreground hover:text-foreground">
                  {selectedIds.size === filteredCheckpoints.length ? 'Deselect' : 'Select'} all
                </button>
              </div>
            )}
            <div className="grid gap-2 sm:grid-cols-2 max-h-[60vh] overflow-y-auto overscroll-contain">
              {filteredCheckpoints.map((cp) => {
                const isActive = checkpoints.activeCheckpoint === cp.name
                const isSelected = selectedIds.has(cp.name)
                return (
                  <div key={cp.name} className={cn(
                    "flex items-center justify-between rounded-lg border p-3 text-sm transition-colors",
                    isActive ? "border-primary/30 bg-primary/[0.08]" : isSelected ? "border-primary/40 bg-primary/5" : "border-border/50 hover:bg-muted/30"
                  )}>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        {checkpoints.checkpoints.length > 2 && (
                          <button
                            onClick={() => toggleSelect(cp.name)}
                            role="checkbox"
                            aria-checked={isSelected}
                            aria-label={isSelected ? `Deselect ${cp.name}` : `Select ${cp.name}`}
                            className={`h-4 w-4 rounded border shrink-0 flex items-center justify-center transition-colors ${isSelected ? 'bg-primary border-primary text-primary-foreground' : 'border-border hover:border-primary/50'}`}
                          >
                            {isSelected && <span className="text-[8px] font-bold" aria-hidden="true">✓</span>}
                          </button>
                        )}
                        <p className="truncate font-medium text-xs">{cp.name}</p>
                        {isActive && <span className="text-primary text-[10px]">✓</span>}
                      </div>
                      {cp.description ? (
                        <p className="text-[11px] text-muted-foreground mt-0.5">{cp.description}</p>
                      ) : (
                        <div className="flex flex-wrap gap-x-2 gap-y-0.5 text-[11px] text-muted-foreground mt-0.5">
                          {cp.loss != null && <span>loss {cp.loss.toFixed(4)}</span>}
                          {cp.epochs_trained != null && <span>{cp.epochs_trained} epochs</span>}
                          {cp.training_duration_s != null && <span>{cp.training_duration_s < 60 ? `${cp.training_duration_s.toFixed(0)}s` : `${Math.floor(cp.training_duration_s / 60)}m ${Math.round(cp.training_duration_s % 60)}s`}</span>}
                          {cp.training_dataset && cp.training_dataset !== 'gpt2-generated' && <span>{cp.training_dataset}</span>}
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-1 shrink-0 ml-2">
                      {!isActive && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-6 text-xs"
                          disabled={loadingCheckpoint === cp.name}
                          onClick={async () => {
                            setLoadingCheckpoint(cp.name)
                            try {
                              await checkpoints.handleLoadCheckpoint(cp.name, addToast)
                            } finally {
                              setLoadingCheckpoint(null)
                            }
                          }}
                        >
                          {loadingCheckpoint === cp.name ? 'Loading...' : 'Load'}
                        </Button>
                      )}
                      <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => onContinue(cp.name)}>Continue</Button>
                      <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={async () => {
                        try {
                          addToast('Downloading checkpoint...', 'info')
                          const blob = await trainingJobsController.downloadCheckpoint(cp.name)
                          downloadBlob(blob, `${cp.name}.sou`)
                          addToast('Checkpoint downloaded', 'success')
                        } catch (e) {
                          addToast(extractErrorMessage(e, 'Download failed'), 'error')
                        }
                      }} aria-label="Download checkpoint">
                        <IconDownload className="h-3 w-3" />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 text-xs text-destructive hover:text-destructive"
                        disabled={loadingCheckpoint === cp.name}
                        onClick={async () => {
                          setLoadingCheckpoint(cp.name)
                          try {
                            await checkpoints.handleDeleteCheckpoint(cp.name, addToast)
                          } finally {
                            setLoadingCheckpoint(null)
                          }
                        }}
                      >
                        {loadingCheckpoint === cp.name ? '...' : 'Del'}
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
