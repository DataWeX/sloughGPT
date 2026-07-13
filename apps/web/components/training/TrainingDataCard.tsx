'use client'

import { useCallback, useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { trainingController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'
import type { TrainingDataStats, TrainingPair } from '@/lib/training-controller'

export function TrainingDataCard() {
  const addToast = useToastStore(s => s.addToast)
  const [stats, setStats] = useState<TrainingDataStats | null>(null)
  const [pairs, setPairs] = useState<TrainingPair[]>([])
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState(false)

  const fetchData = useCallback(async () => {
    try {
      const [s, p] = await Promise.all([
        trainingController.getTrainingStats(),
        trainingController.getPendingPairs(20),
      ])
      setStats(s)
      setPairs(p.pairs)
    } catch {
      // might not be available
    } finally {
      setLoading(false)
    }
  }, [])

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

  const handleDeleteSynced = useCallback(async () => {
    try {
      const result = await trainingController.deleteSyncedPairs()
      addToast(`Deleted ${result.count} synced pairs`, 'success')
      void fetchData()
    } catch {
      addToast('Delete failed', 'error')
    }
  }, [addToast, fetchData])

  if (loading) {
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

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Training data</CardTitle>
        <div className="flex items-center gap-2">
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

        {/* Pairs list */}
        {pairs.length > 0 && (
          <div className="space-y-2">
            <button
              className="text-[11px] text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? 'Hide' : 'Show'} recent pairs ({pairs.length})
            </button>
            {expanded && (
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
