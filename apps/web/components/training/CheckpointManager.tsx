'use client'

import { useState, useCallback, memo } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, cn, Checkbox } from '@sloughgpt/strui'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@sloughgpt/strui'
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from '@sloughgpt/strui'
import { trainingJobsController } from '@/lib/training-controller'
import type { Checkpoint } from '@/lib/souls-controller'

const PAGE_SIZE = 10

interface Props {
  checkpoints: Checkpoint[]
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
  onRefresh: () => void
}

export const CheckpointManager = memo(function CheckpointManager({ checkpoints, addToast, onRefresh }: Props) {
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [page, setPage] = useState(0)
  const [info, setInfo] = useState<Record<string, unknown> | null>(null)
  const [infoName, setInfoName] = useState('')
  const [pendingDelete, setPendingDelete] = useState<string[]>([])

  const toggle = useCallback((name: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name); else next.add(name)
      return next
    })
  }, [])

  const toggleAll = useCallback(() => {
    setSelected(prev => prev.size === checkpoints.length ? new Set() : new Set(checkpoints.map(c => c.name)))
  }, [checkpoints])

  const handleLoad = useCallback(async (name: string) => {
    try { await trainingJobsController.loadCheckpoint(name); addToast(`Loaded: ${name}`, 'success') }
    catch { addToast('Could not load checkpoint', 'error') }
  }, [addToast])

  const handleDelete = useCallback(async (name: string) => {
    try { await trainingJobsController.deleteCheckpoint(name); addToast(`Deleted: ${name}`, 'success'); onRefresh() }
    catch { addToast('Could not delete checkpoint', 'error') }
  }, [addToast, onRefresh])

  const handleBatchDelete = useCallback(async () => {
    if (selected.size === 0) return
    setPendingDelete(Array.from(selected))
  }, [selected])

  const confirmBatchDelete = useCallback(async () => {
    try { await trainingJobsController.deleteCheckpointsBatch(pendingDelete); addToast(`Deleted ${pendingDelete.length} checkpoints`, 'success'); setSelected(new Set()); onRefresh() }
    catch { addToast('Batch delete failed', 'error') }
    setPendingDelete([])
  }, [pendingDelete, addToast, onRefresh])

  const handleDownload = useCallback(async (name: string) => {
    try {
      const blob = await trainingJobsController.downloadCheckpoint(name)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a'); a.href = url; a.download = name; a.click()
      URL.revokeObjectURL(url)
    } catch { addToast('Download failed', 'error') }
  }, [addToast])

  const handleInfo = useCallback(async (name: string) => {
    setInfoName(name)
    try { setInfo(await trainingJobsController.getCheckpointInfo(name)) }
    catch { addToast('Could not load info', 'error'); setInfo(null) }
  }, [addToast])

  const start = page * PAGE_SIZE
  const pageItems = checkpoints.slice(start, start + PAGE_SIZE)
  const totalPages = Math.ceil(checkpoints.length / PAGE_SIZE)

  if (checkpoints.length === 0) return null

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Checkpoints ({checkpoints.length})</CardTitle>
          <div className="flex gap-1">
            {selected.size > 0 && (
              <Button size="sm" variant="ghost" className="text-destructive text-xs" onClick={handleBatchDelete}>
                Delete {selected.size}
              </Button>
            )}
            <Button size="sm" variant="ghost" className="text-xs" onClick={toggleAll}>
              {selected.size === checkpoints.length ? 'Deselect all' : 'Select all'}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {pageItems.map(c => (
            <div key={c.name} className={cn('flex items-center justify-between rounded border p-3 text-sm', selected.has(c.name) && 'border-primary bg-primary/5')}>
              <div className="min-w-0 flex-1 flex items-center gap-2">
                <Checkbox checked={selected.has(c.name)} onCheckedChange={() => toggle(c.name)} aria-label={`Select ${c.name}`} className="h-3.5 w-3.5 rounded border-border" />
                <div className="min-w-0">
                  <p className="truncate font-medium">{c.name}</p>
                </div>
                <div className="flex gap-3 text-xs text-muted-foreground">
                  {c.loss != null && <span>Loss {c.loss.toFixed(4)}</span>}
                  {c.steps != null && <span>{c.steps} steps</span>}
                  {c.epochs != null && <span>{c.epochs} epochs</span>}
                  {c.size_mb != null && <span>{c.size_mb.toFixed(1)} MB</span>}
                </div>
              </div>
              <div className="flex items-center gap-1">
                <Button size="sm" variant="ghost" onClick={() => void handleInfo(c.name)}>Info</Button>
                <Button size="sm" variant="ghost" onClick={() => void handleDownload(c.name)}>Download</Button>
                <Button size="sm" variant="ghost" onClick={() => void handleLoad(c.name)}>Load</Button>
                <Button size="sm" variant="ghost" className="text-destructive" onClick={() => void handleDelete(c.name)}>Delete</Button>
              </div>
            </div>
          ))}
        </div>
        {totalPages > 1 && (
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-border/30">
            <span className="text-xs text-muted-foreground">{start + 1}–{Math.min(start + PAGE_SIZE, checkpoints.length)} of {checkpoints.length}</span>
            <div className="flex gap-1">
              <Button size="sm" variant="ghost" className="text-xs" disabled={page === 0} onClick={() => setPage(p => p - 1)}>Prev</Button>
              <Button size="sm" variant="ghost" className="text-xs" disabled={page + 1 >= totalPages} onClick={() => setPage(p => p + 1)}>Next</Button>
            </div>
          </div>
        )}
      </CardContent>

      <Dialog open={!!info} onOpenChange={(open) => { if (!open) setInfo(null) }}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="text-base">{infoName}</DialogTitle>
          </DialogHeader>
          {info && (
            <div className="space-y-1.5 text-xs max-h-[60vh] overflow-auto">
              {Object.entries(info).map(([k, v]) => (
                <div key={k} className="flex justify-between border-b border-border/30 py-1">
                  <span className="text-muted-foreground">{k}</span>
                  <span className="font-mono text-right">{typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>
                </div>
              ))}
            </div>
          )}
        </DialogContent>
      </Dialog>

      <AlertDialog open={pendingDelete.length > 0} onOpenChange={(open) => { if (!open) setPendingDelete([]) }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete {pendingDelete.length} checkpoints</AlertDialogTitle>
            <AlertDialogDescription>This action cannot be undone.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => void confirmBatchDelete()}>Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  )
})
