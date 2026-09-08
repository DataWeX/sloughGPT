'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { ActionCard, Button, Skeleton } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { trainingJobsController, type TrainingBuild } from '@/lib/training-controller'
import { soulsController } from '@/lib/souls-controller'
import { ConfirmDialog } from '@/components/ConfirmDialog'

interface Props {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}

const BUILD_TYPE_LABELS: Record<string, string> = {
  'auto-train': 'auto-train',
  'lora': 'lora',
  'hf-finetune': 'hf-finetune',
  'hf-finetuned-dir': 'hf-dir',
  'vlm': 'vlm',
  'visual': 'visual',
}

export function TrainingBuildsCard({ addToast }: Props) {
  const [builds, setBuilds] = useState<TrainingBuild[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [loadingModel, setLoadingModel] = useState<string | null>(null)
  const [filter, setFilter] = useState<string>('all')
  const [page, setPage] = useState(0)
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  const PAGE_SIZE = 10

  const activeRef = useRef(true)

  const fetchBuilds = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await trainingJobsController.listBuilds()
      if (activeRef.current) setBuilds(result ?? [])
    } catch {
      if (activeRef.current) { setBuilds([]); setError('Could not load builds') }
    } finally {
      if (activeRef.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    activeRef.current = true
    void fetchBuilds()
    return () => { activeRef.current = false }
  }, [fetchBuilds])

  const handleLoad = useCallback(async (name: string, buildType: string) => {
    setLoadingModel(name)
    try {
      if (buildType === 'hf-finetune' || buildType === 'hf-finetuned-dir') {
        await trainingJobsController.loadFineTuned(name)
      } else {
        await soulsController.loadCheckpoint(name)
      }
      addToast(`Loaded: ${name}`, 'success')
    } catch {
      addToast('Could not load checkpoint', 'error')
    } finally {
      setLoadingModel(null)
    }
  }, [addToast])

  const handleDelete = useCallback(async (name: string) => {
    try {
      await trainingJobsController.deleteCheckpoint(name)
      addToast(`Deleted: ${name}`, 'success')
      void fetchBuilds()
    } catch {
      addToast('Could not delete', 'error')
    }
  }, [addToast, fetchBuilds])

  const handleDownload = useCallback(async (name: string) => {
    try {
      const blob = await trainingJobsController.downloadCheckpoint(name)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = name.endsWith('.soul') ? name : `${name}.soul`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      addToast(`Downloaded: ${name}`, 'success')
    } catch {
      addToast('Could not download', 'error')
    }
  }, [addToast])

  const filtered = filter === 'all' ? builds : builds.filter(b => b.build_type === filter)
  const typeCounts = builds.reduce((acc, b) => { acc[b.build_type] = (acc[b.build_type] || 0) + 1; return acc }, {} as Record<string, number>)

  return (
    <ActionCard
      title={`Builds (${builds.length})`}
      actions={<Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => void fetchBuilds()}>Refresh</Button>}
      testId="training-builds"
      contentClassName="space-y-2"
    >
        {loading ? (
          <div className="space-y-1.5">
            <Skeleton className="h-3 w-28" />
            <Skeleton className="h-14 w-full" />
            <Skeleton className="h-14 w-full" />
          </div>
        ) : error ? (
          <StatusBanner variant="error" message={error} dismissible={false} onRetry={() => void fetchBuilds()} />
        ) : builds.length === 0 ? (
          <p className="text-[10px] text-muted-foreground/60">No builds found. Start training to create builds.</p>
        ) : (
          <>
            <div className="flex flex-wrap gap-0.5">
              <Button size="sm" variant={filter === 'all' ? 'default' : 'ghost'} className="h-6 text-[10px]" onClick={() => { setFilter('all'); setPage(0) }}>
                All ({builds.length})
              </Button>
              {Object.entries(typeCounts).map(([type, count]) => (
                <Button key={type} size="sm" variant={filter === type ? 'default' : 'ghost'} className="h-6 text-[10px]" onClick={() => { setFilter(type); setPage(0) }}>
                  {BUILD_TYPE_LABELS[type] ?? type} ({count})
                </Button>
              ))}
            </div>

            <div className="space-y-1">
              {filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map(b => (
                <div key={b.name} className="flex items-center justify-between rounded-lg border border-border/40 p-2 hover:bg-muted/20 transition-colors">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <p className="truncate font-medium text-[11px]">{b.name}</p>
                      <span className="rounded-full bg-muted/50 px-1 py-px text-[8px] text-muted-foreground/60">
                        {BUILD_TYPE_LABELS[b.build_type] ?? b.build_type}
                      </span>
                    </div>
                    <div className="flex gap-1.5 text-[9px] text-muted-foreground/60 mt-0.5 tabular-nums">
                      {b.loss != null && <span>Loss {b.loss.toFixed(4)}</span>}
                      {b.epochs != null && <span>{b.epochs} epochs</span>}
                      {b.size_mb != null && <span>{b.size_mb.toFixed(1)} MB</span>}
                      {b.model && <span>Model: {b.model}</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-0.5">
                    <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => void handleLoad(b.name, b.build_type)} disabled={loadingModel === b.name}>
                      {loadingModel === b.name ? 'Loading...' : 'Load'}
                    </Button>
                    <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => void handleDownload(b.name)}>Download</Button>
                    <Button size="sm" variant="ghost" className="h-6 text-[10px] text-destructive" onClick={() => setPendingDelete(b.name)}>Delete</Button>
                  </div>
                </div>
              ))}
            </div>
            {filtered.length > PAGE_SIZE && (
              <div className="flex items-center justify-between mt-2 pt-2 border-t border-border/30">
                <span className="text-[9px] text-muted-foreground/60 tabular-nums">
                  {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, filtered.length)} of {filtered.length}
                </span>
                <div className="flex gap-0.5">
                  <Button size="sm" variant="ghost" className="h-6 text-[10px]" disabled={page === 0} onClick={() => setPage(p => p - 1)}>Prev</Button>
                  <Button size="sm" variant="ghost" className="h-6 text-[10px]" disabled={(page + 1) * PAGE_SIZE >= filtered.length} onClick={() => setPage(p => p + 1)}>Next</Button>
                </div>
              </div>
            )}
          </>
        )}
        <ConfirmDialog
          open={pendingDelete !== null}
          onOpenChange={(open) => { if (!open) setPendingDelete(null) }}
          title="Delete this build?"
          description={`"${pendingDelete}" will be permanently removed. This cannot be undone.`}
          confirmLabel="Delete"
          onConfirm={() => { if (pendingDelete) void handleDelete(pendingDelete); setPendingDelete(null) }}
        />
    </ActionCard>
  )
}
