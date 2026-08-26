'use client'

import { useState, useEffect, useCallback, memo } from 'react'
import { cn, Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import { trainingJobsController, type TrainingJob } from '@/lib/training-controller'
import { downloadJson } from '@/lib/download-utils'

interface Props {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span className={cn('text-[10px] px-1.5 py-0.5 rounded font-medium', status === 'completed' ? 'bg-success/15 text-success' :
      status === 'running' ? 'bg-warning/15 text-warning' :
      status === 'failed' ? 'bg-destructive/15 text-destructive' :
      'bg-muted text-muted-foreground')}>{status}</span>
  )
}

function formatDuration(seconds?: number | null): string {
  if (seconds == null) return '—'
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}

function formatDate(iso?: string | null): string {
  if (!iso) return '—'
  try {
    const d = new Date(iso)
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  } catch { return iso }
}

export const TrainingHistoryView = memo(function TrainingHistoryView({ addToast }: Props) {
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<string>('all')
  const [expanded, setExpanded] = useState<string | null>(null)
  const [page, setPage] = useState(0)
  const PAGE_SIZE = 20

  const fetchJobs = useCallback(async () => {
    setLoading(true)
    try {
      const result = await trainingJobsController.list()
      setJobs(result ?? [])
    } catch {
      addToast('Could not fetch training history', 'error')
      setJobs([])
    } finally {
      setLoading(false)
    }
  }, [addToast])

  const handleExport = useCallback(() => {
    if (jobs.length === 0) return
    const exportData = jobs.map(j => ({
      id: j.id,
      name: j.name,
      status: j.status,
      method: j.method,
      created_at: j.created_at,
      finished_at: j.finished_at,
      loss: j.loss,
      train_loss: j.train_loss,
      eval_loss: j.eval_loss,
      epochs: j.epochs,
      epochs_completed: j.epochs_completed,
      elapsed_s: j.elapsed_s,
      checkpoint: j.checkpoint,
      dataset: j.dataset,
      model: j.model,
      error: j.error,
    }))
    downloadJson(exportData, `training-history-${new Date().toISOString().slice(0, 10)}.json`)
    addToast('Training history exported', 'success')
  }, [jobs, addToast])

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      try {
        const result = await trainingJobsController.list()
        if (active) setJobs(result ?? [])
      } catch {
        if (active) {
          addToast('Could not fetch training history', 'error')
          setJobs([])
        }
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [addToast])

  const filtered = filter === 'all' ? jobs : jobs.filter(j => j.status === filter)
  const statusCounts = jobs.reduce((acc, j) => { acc[j.status] = (acc[j.status] || 0) + 1; return acc }, {} as Record<string, number>)

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Training History ({jobs.length})</CardTitle>
          <div className="flex items-center gap-1">
            {jobs.length > 0 && (
              <Button size="sm" variant="ghost" onClick={handleExport}>Export</Button>
            )}
            <Button size="sm" variant="ghost" onClick={() => void fetchJobs()}>Refresh</Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {loading ? (
          <p className="text-xs text-muted-foreground">Loading...</p>
        ) : jobs.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            No training jobs yet. Start training to see history.
          </p>
        ) : (
          <>
            <div className="flex flex-wrap gap-1">
              <Button size="sm" variant={filter === 'all' ? 'default' : 'ghost'} onClick={() => { setFilter('all'); setPage(0) }}>
                All ({jobs.length})
              </Button>
              {Object.entries(statusCounts).map(([status, count]) => (
                <Button key={status} size="sm" variant={filter === status ? 'default' : 'ghost'} onClick={() => { setFilter(status); setPage(0) }}>
                  {status} ({count})
                </Button>
              ))}
            </div>

            <div className="space-y-1">
              {filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map(job => (
                <div
                  key={job.id}
                  className="rounded border p-2.5 text-xs hover:bg-muted/30 transition-colors cursor-pointer"
                  onClick={() => setExpanded(expanded === job.id ? null : job.id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 min-w-0">
                      <StatusBadge status={job.status} />
                      <span className="truncate font-medium">{job.name || job.id}</span>
                    </div>
                    <div className="flex items-center gap-3 text-muted-foreground shrink-0 ml-2">
                      {job.method && <span className="text-[10px] bg-muted px-1 py-0.5 rounded">{job.method}</span>}
                      <span>{formatDate(job.created_at)}</span>
                    </div>
                  </div>

                  {expanded === job.id && (
                    <div className="mt-2 pt-2 border-t border-border/30 grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-muted-foreground">
                      {job.loss != null && <div>Loss: {job.loss.toFixed(4)}</div>}
                      {job.epochs != null && <div>Epochs: {job.epochs}</div>}
                      {job.elapsed_s != null && <div>Duration: {formatDuration(job.elapsed_s)}</div>}
                      {job.checkpoint && <div>Checkpoint: {job.checkpoint}</div>}
                      {job.dataset && <div>Dataset: {job.dataset}</div>}
                      {job.model && <div>Model: {job.model}</div>}
                      {job.error && <div className="text-destructive">Error: {job.error}</div>}
                    </div>
                  )}
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
})
