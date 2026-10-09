'use client'

import { useState, useEffect, useCallback, memo, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { trainingJobsController, type TrainingPair, type TrainingDataStats } from '@/lib/training-controller'

interface Props {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}

export const TrainingDataCard = memo(function TrainingDataCard({ addToast }: Props) {
  const [stats, setStats] = useState<TrainingDataStats | null>(null)
  const [pairs, setPairs] = useState<TrainingPair[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const [total, setTotal] = useState(0)
  const limit = 20

  const activeRef = useRef(true)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [statsResult, pairsResult] = await Promise.all([
        trainingJobsController.getTrainingStats(),
        trainingJobsController.listTrainingPairs({ limit, offset: page * limit, search: search || undefined }),
      ])
      if (activeRef.current) {
        setStats(statsResult)
        setPairs(pairsResult.pairs ?? [])
        setTotal(pairsResult.total ?? 0)
      }
    } catch {
      if (activeRef.current) {
        addToast('Failed to load training data', 'error')
        setError('Could not load training data')
        setStats(null)
        setPairs([])
      }
    } finally {
      if (activeRef.current) setLoading(false)
    }
  }, [page, search, addToast])

  useEffect(() => {
    activeRef.current = true
    void fetchData()
    return () => { activeRef.current = false }
  }, [fetchData])

  const handleSearch = useCallback(() => {
    setPage(0)
    void fetchData()
  }, [fetchData])

  const handleDelete = useCallback(async (id: string) => {
    try {
      await trainingJobsController.deletePair(id)
      addToast('Pair deleted', 'success')
      void fetchData()
    } catch {
      addToast('Could not delete pair', 'error')
    }
  }, [addToast, fetchData])

  const handleUpdateQuality = useCallback(async (id: string, quality: number) => {
    try {
      await trainingJobsController.updatePairQuality(id, quality)
      addToast('Quality updated', 'success')
      void fetchData()
    } catch {
      addToast('Could not update quality', 'error')
    }
  }, [addToast, fetchData])

  const handleDeleteSynced = useCallback(async () => {
    try {
      const result = await trainingJobsController.deleteSyncedPairs()
      addToast(`Deleted ${result.count} synced pairs`, 'success')
      void fetchData()
    } catch {
      addToast('Could not delete synced pairs', 'error')
    }
  }, [addToast, fetchData])

  const totalPages = Math.ceil(total / limit)

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Training data ({total})</CardTitle>
          <div className="flex gap-2">
            {stats && stats.synced > 0 && (
              <Button size="sm" variant="ghost" className="text-destructive" onClick={handleDeleteSynced} aria-label="Delete all synced training data">
                Delete synced ({stats.synced})
              </Button>
            )}
            <Button size="sm" variant="ghost" onClick={() => void fetchData()} aria-label="Refresh training data">Refresh</Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {loading && !stats ? (
          <p className="text-xs text-muted-foreground">Loading...</p>
        ) : error ? (
          <StatusBanner variant="error" message={error} dismissible={false} onRetry={() => void fetchData()} />
        ) : stats ? (
          <div className="grid grid-cols-4 gap-2">
            <div className="rounded-lg bg-muted/20 px-2 py-1.5 text-center">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Total</p>
              <p className="text-sm font-semibold tabular-nums mt-0.5">{stats.total}</p>
            </div>
            <div className="rounded-lg bg-muted/20 px-2 py-1.5 text-center">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Pending</p>
              <p className="text-sm font-semibold tabular-nums mt-0.5">{stats.pending}</p>
            </div>
            <div className="rounded-lg bg-muted/20 px-2 py-1.5 text-center">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Synced</p>
              <p className="text-sm font-semibold tabular-nums mt-0.5">{stats.synced}</p>
            </div>
            <div className="rounded-lg bg-muted/20 px-2 py-1.5 text-center">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Used</p>
              <p className="text-sm font-semibold tabular-nums mt-0.5">{stats.used}</p>
            </div>
          </div>
        ) : null}

        <div className="flex gap-2">
          <Input
            placeholder="Search pairs..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSearch()}
            className="h-8 text-xs"
          />
          <Button size="sm" variant="ghost" onClick={handleSearch} aria-label="Search training pairs">Search</Button>
        </div>

        {loading ? (
          <p className="text-xs text-muted-foreground">Loading pairs...</p>
        ) : pairs.length === 0 ? (
          <p className="text-xs text-muted-foreground">No training pairs found.</p>
        ) : (
          <div className="space-y-1.5">
            {pairs.map(p => (
              <div key={p.id} className="rounded-lg border border-border/40 p-2.5 hover:bg-muted/20 transition-colors">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1 space-y-0.5">
                    <p className="text-[11px] text-muted-foreground/60 truncate">User: {p.user_msg}</p>
                    <p className="text-xs truncate">{p.assistant_msg}</p>
                    <div className="flex gap-2 text-[10px] text-muted-foreground/50">
                      <span>Quality: {p.quality}</span>
                      <span>{new Date(p.timestamp).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Select value={String(p.quality)} onValueChange={v => void handleUpdateQuality(p.id, Number(v))}>
                      <SelectTrigger className="h-6 w-12 text-[10px]" aria-label="Quality rating">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {[1, 2, 3, 4, 5].map(q => <SelectItem key={q} value={String(q)}>{q}</SelectItem>)}
                      </SelectContent>
                    </Select>
                    <Button size="sm" variant="ghost" className="text-destructive h-6 text-[10px]" onClick={() => void handleDelete(p.id)} aria-label={`Delete training pair ${p.id.slice(0, 8)}`}>
                      Delete
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {totalPages > 1 && (
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Page {page + 1} of {totalPages}</span>
            <div className="flex gap-1">
              <Button size="sm" variant="ghost" disabled={page === 0} onClick={() => setPage(p => p - 1)} aria-label="Previous page">Prev</Button>
              <Button size="sm" variant="ghost" disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)} aria-label="Next page">Next</Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
})
