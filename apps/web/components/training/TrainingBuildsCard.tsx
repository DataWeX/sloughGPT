'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Skeleton } from '@sloughgpt/strui'
import { trainingJobsController, type TrainingBuild } from '@/lib/training-controller'
import { soulsController } from '@/lib/souls-controller'

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
  const [loadingModel, setLoadingModel] = useState<string | null>(null)
  const [filter, setFilter] = useState<string>('all')
  const [page, setPage] = useState(0)
  const PAGE_SIZE = 10

  const fetchBuilds = useCallback(async () => {
    setLoading(true)
    try {
      const result = await trainingJobsController.listBuilds()
      setBuilds(result ?? [])
    } catch {
      setBuilds([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      try {
        const result = await trainingJobsController.listBuilds()
        if (active) setBuilds(result ?? [])
      } catch {
        if (active) setBuilds([])
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [])

  const handleLoad = useCallback(async (name: string) => {
    setLoadingModel(name)
    try {
      await soulsController.loadCheckpoint(name)
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
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Builds ({builds.length})</CardTitle>
          <Button size="sm" variant="ghost" onClick={() => void fetchBuilds()}>Refresh</Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {loading ? (
          <div className="space-y-2">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        ) : builds.length === 0 ? (
          <p className="text-xs text-muted-foreground">No builds found. Start training to create builds.</p>
        ) : (
          <>
            <div className="flex flex-wrap gap-1">
              <Button size="sm" variant={filter === 'all' ? 'default' : 'ghost'} onClick={() => { setFilter('all'); setPage(0) }}>
                All ({builds.length})
              </Button>
              {Object.entries(typeCounts).map(([type, count]) => (
                <Button key={type} size="sm" variant={filter === type ? 'default' : 'ghost'} onClick={() => { setFilter(type); setPage(0) }}>
                  {BUILD_TYPE_LABELS[type] ?? type} ({count})
                </Button>
              ))}
            </div>

            <div className="space-y-2">
              {filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map(b => (
                <div key={b.name} className="flex items-center justify-between rounded border p-3 text-sm">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <p className="truncate font-medium">{b.name}</p>
                      <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                        {BUILD_TYPE_LABELS[b.build_type] ?? b.build_type}
                      </span>
                    </div>
                    <div className="flex gap-3 text-xs text-muted-foreground">
                      {b.loss != null && <span>Loss {b.loss.toFixed(4)}</span>}
                      {b.epochs != null && <span>{b.epochs} epochs</span>}
                      {b.size_mb != null && <span>{b.size_mb.toFixed(1)} MB</span>}
                      {b.model && <span>Model: {b.model}</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button size="sm" variant="ghost" onClick={() => void handleLoad(b.name)} disabled={loadingModel === b.name}>
                      {loadingModel === b.name ? 'Loading...' : 'Load'}
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => void handleDownload(b.name)}>Download</Button>
                    <Button size="sm" variant="ghost" className="text-destructive" onClick={() => void handleDelete(b.name)}>Delete</Button>
                  </div>
                </div>
              ))}
            </div>
            {filtered.length > PAGE_SIZE && (
              <div className="flex items-center justify-between mt-3 pt-3 border-t border-border/30">
                <span className="text-[10px] text-muted-foreground">
                  {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, filtered.length)} of {filtered.length}
                </span>
                <div className="flex gap-1">
                  <Button size="sm" variant="ghost" className="text-[10px]" disabled={page === 0} onClick={() => setPage(p => p - 1)}>Prev</Button>
                  <Button size="sm" variant="ghost" className="text-[10px]" disabled={(page + 1) * PAGE_SIZE >= filtered.length} onClick={() => setPage(p => p + 1)}>Next</Button>
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}
