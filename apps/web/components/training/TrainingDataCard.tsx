'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Input } from '@sloughgpt/strui'
import { trainingJobsController, type TrainingPair, type TrainingDataStats } from '@/lib/training-controller'

interface Props {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}

export function TrainingDataCard({ addToast }: Props) {
  const [stats, setStats] = useState<TrainingDataStats | null>(null)
  const [pairs, setPairs] = useState<TrainingPair[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const [total, setTotal] = useState(0)
  const limit = 20

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [statsResult, pairsResult] = await Promise.all([
        trainingJobsController.getTrainingStats(),
        trainingJobsController.listTrainingPairs({ limit, offset: page * limit, search: search || undefined }),
      ])
      setStats(statsResult)
      setPairs(pairsResult.pairs ?? [])
      setTotal(pairsResult.total ?? 0)
    } catch {
      addToast('Failed to load training data', 'error')
      setError('Could not load training data')
      setStats(null)
      setPairs([])
    } finally {
      setLoading(false)
    }
  }, [page, search, addToast])

  useEffect(() => { void fetchData() }, [fetchData])

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
              <Button size="sm" variant="ghost" className="text-destructive" onClick={handleDeleteSynced}>
                Delete synced ({stats.synced})
              </Button>
            )}
            <Button size="sm" variant="ghost" onClick={() => void fetchData()}>Refresh</Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {loading && !stats ? (
          <p className="text-xs text-muted-foreground">Loading...</p>
        ) : error ? (
          <div className="text-center py-4">
            <p className="text-sm text-destructive mb-2">{error}</p>
            <Button size="sm" variant="ghost" onClick={() => void fetchData()}>Retry</Button>
          </div>
        ) : stats ? (
          <div className="grid grid-cols-4 gap-2 text-center text-xs">
            <div className="rounded bg-muted/30 p-2">
              <p className="text-muted-foreground">Total</p>
              <p className="font-medium">{stats.total}</p>
            </div>
            <div className="rounded bg-muted/30 p-2">
              <p className="text-muted-foreground">Pending</p>
              <p className="font-medium">{stats.pending}</p>
            </div>
            <div className="rounded bg-muted/30 p-2">
              <p className="text-muted-foreground">Synced</p>
              <p className="font-medium">{stats.synced}</p>
            </div>
            <div className="rounded bg-muted/30 p-2">
              <p className="text-muted-foreground">Used</p>
              <p className="font-medium">{stats.used}</p>
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
          <Button size="sm" variant="ghost" onClick={handleSearch}>Search</Button>
        </div>

        {loading ? (
          <p className="text-xs text-muted-foreground">Loading pairs...</p>
        ) : pairs.length === 0 ? (
          <p className="text-xs text-muted-foreground">No training pairs found.</p>
        ) : (
          <div className="space-y-2">
            {pairs.map(p => (
              <div key={p.id} className="rounded border p-3 text-sm">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1 space-y-1">
                    <p className="text-xs text-muted-foreground truncate">User: {p.user_msg}</p>
                    <p className="text-xs truncate">Assistant: {p.assistant_msg}</p>
                    <div className="flex gap-2 text-[10px] text-muted-foreground">
                      <span>Quality: {p.quality}</span>
                      <span>{new Date(p.timestamp).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <select
                      value={p.quality}
                      onChange={e => void handleUpdateQuality(p.id, Number(e.target.value))}
                      className="h-7 rounded border bg-background px-1 text-xs"
                    >
                      {[1, 2, 3, 4, 5].map(q => <option key={q} value={q}>{q}</option>)}
                    </select>
                    <Button size="sm" variant="ghost" className="text-destructive" onClick={() => void handleDelete(p.id)}>
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
              <Button size="sm" variant="ghost" disabled={page === 0} onClick={() => setPage(p => p - 1)}>Prev</Button>
              <Button size="sm" variant="ghost" disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}>Next</Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
