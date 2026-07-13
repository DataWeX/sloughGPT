'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { trainingController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'
import type { TrainingDataStats, TrainingPair } from '@/lib/training-controller'

export function TrainingDataCard() {
  const addToast = useToastStore(s => s.addToast)
  const [stats, setStats] = useState<TrainingDataStats | null>(null)
  const [pairs, setPairs] = useState<TrainingPair[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState(false)
  const [search, setSearch] = useState('')
  const [qualityFilter, setQualityFilter] = useState<string>('all')
  const [sessionFilter, setSessionFilter] = useState<string>('all')
  const [allSessions, setAllSessions] = useState<string[]>([])
  const [page, setPage] = useState(0)
  const pageSize = 20

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const minQuality = qualityFilter === 'all' ? undefined
        : qualityFilter === 'high' ? 4
        : qualityFilter === 'mid' ? 2
        : 0
      const [s, p] = await Promise.all([
        trainingController.getTrainingStats(),
        trainingController.listTrainingPairs({
          limit: pageSize,
          offset: page * pageSize,
          min_quality: minQuality,
          session_id: sessionFilter === 'all' ? undefined : sessionFilter,
          search: search || undefined,
        }),
      ])
      setStats(s)
      setPairs(p.pairs)
      setTotal(p.total)
      // Collect unique session IDs from all pairs for filter dropdown
      const sessionIds = new Set(allSessions)
      p.pairs.forEach(pair => { if (pair.session_id) sessionIds.add(pair.session_id) })
      setAllSessions(Array.from(sessionIds).sort())
    } catch {
      // might not be available
    } finally {
      setLoading(false)
    }
  }, [page, qualityFilter, sessionFilter, search]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { void fetchData() }, [fetchData])

  const handleDelete = useCallback(async (id: string) => {
    try {
      await trainingController.deletePair(id)
      setPairs(prev => prev.filter(p => p.id !== id))
      setStats(prev => prev ? { ...prev, total: prev.total - 1, pending: prev.pending - 1 } : prev)
      addToast('Pair deleted', 'success')
    } catch {
      addToast('Delete failed', 'error')
    }
  }, [addToast])

  const handleExport = useCallback(async () => {
    try {
      const blob = await trainingController.exportTrainingPairs({
        min_quality: qualityFilter === 'all' ? undefined
          : qualityFilter === 'high' ? 4
          : qualityFilter === 'mid' ? 2
          : 0,
        session_id: sessionFilter === 'all' ? undefined : sessionFilter,
        limit: 5000,
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'training_pairs.jsonl'
      a.click()
      URL.revokeObjectURL(url)
      addToast('Exported training pairs', 'success')
    } catch {
      addToast('Export failed', 'error')
    }
  }, [addToast, qualityFilter, sessionFilter])

  const handleDeleteSynced = useCallback(async () => {
    try {
      const result = await trainingController.deleteSyncedPairs()
      addToast(`Deleted ${result.count} synced pairs`, 'success')
      void fetchData()
    } catch {
      addToast('Delete failed', 'error')
    }
  }, [addToast, fetchData])

  if (loading && !stats) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Training data</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-20 w-full" />
        </CardContent>
      </Card>
    )
  }

  if (!stats) return null

  const totalPages = Math.ceil(total / pageSize)

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Training data</CardTitle>
        <div className="flex items-center gap-2">
          {stats.total > 0 && (
            <Button size="sm" variant="ghost" className="h-6 text-[11px]" onClick={() => void handleExport()}>
              Export JSONL
            </Button>
          )}
          {stats.synced > 0 && (
            <Button size="sm" variant="ghost" className="h-6 text-[11px] text-destructive hover:text-destructive" onClick={() => void handleDeleteSynced()}>
              Clear synced ({stats.synced})
            </Button>
          )}
          <Button size="sm" variant="ghost" className="h-6 text-[11px]" onClick={() => void fetchData()}>
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Stats row */}
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground/70">
          <span>Total: {stats.total}</span>
          <span>Pending: {stats.pending}</span>
          <span>Synced: {stats.synced}</span>
          <span>Used: {stats.used}</span>
        </div>

        {/* Quality breakdown */}
        {Object.keys(stats.by_quality).length > 0 && (
          <div className="flex flex-wrap gap-1">
            {Object.entries(stats.by_quality).sort(([a], [b]) => Number(a) - Number(b)).map(([q, count]) => (
              <span key={q} className="inline-flex items-center gap-1 rounded-full border border-border/40 px-2 py-0.5 text-[10px] text-muted-foreground/70">
                {Number(q).toFixed(1)}: {count}
              </span>
            ))}
          </div>
        )}

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-2">
          <Input
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(0) }}
            placeholder="Search pairs..."
            className="h-7 text-[11px] max-w-[200px]"
          />
          <div className="flex gap-1">
            {(['all', 'high', 'mid', 'low'] as const).map(q => (
              <button
                key={q}
                className={`px-2 py-0.5 rounded text-[10px] transition-colors ${
                  qualityFilter === q
                    ? 'bg-primary/10 text-primary font-medium'
                    : 'text-muted-foreground/70 hover:text-foreground'
                }`}
                onClick={() => { setQualityFilter(q); setPage(0) }}
              >
                {q === 'all' ? 'All' : q === 'high' ? 'Q≥4' : q === 'mid' ? 'Q≥2' : 'Q<2'}
              </button>
            ))}
          </div>
          {allSessions.length > 1 && (
            <select
              value={sessionFilter}
              onChange={e => { setSessionFilter(e.target.value); setPage(0) }}
              className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] text-foreground"
            >
              <option value="all">All sessions</option>
              {allSessions.map(sid => (
                <option key={sid} value={sid}>{sid.slice(0, 12)}...</option>
              ))}
            </select>
          )}
        </div>

        {/* Pairs list */}
        {pairs.length > 0 && (
          <div className="space-y-2">
            <button
              className="text-[11px] text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? 'Hide' : 'Show'} pairs ({total} total)
            </button>
            {expanded && (
              <>
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {pairs.map(pair => (
                    <div key={pair.id} className="rounded-lg border border-border/40 p-2 text-[11px] space-y-1">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0 flex-1">
                          <p className="text-foreground font-medium truncate">{pair.user_msg}</p>
                          <p className="text-muted-foreground/70 truncate">{pair.assistant_msg}</p>
                        </div>
                        <div className="flex items-center gap-1 shrink-0">
                          <span className="text-[10px] text-muted-foreground/50">Q:{pair.quality.toFixed(1)}</span>
                          <button
                            className="text-muted-foreground/40 hover:text-destructive transition-colors"
                            onClick={() => void handleDelete(pair.id)}
                            aria-label="Delete pair"
                          >
                            x
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-muted-foreground/50">
                      Page {page + 1} of {totalPages}
                    </span>
                    <div className="flex gap-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-5 text-[10px]"
                        disabled={page === 0}
                        onClick={() => setPage(p => p - 1)}
                      >
                        Prev
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-5 text-[10px]"
                        disabled={page >= totalPages - 1}
                        onClick={() => setPage(p => p + 1)}
                      >
                        Next
                      </Button>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {pairs.length === 0 && stats.total === 0 && (
          <p className="text-xs text-muted-foreground/50">
            No training data yet. Train from conversations or chat to generate pairs.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
