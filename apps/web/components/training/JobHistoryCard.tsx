'use client'

import { useRouter } from 'next/navigation'
import { useMemo, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { trainingController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'
import { downloadJson } from '@/lib/download-utils'
import { todayDateString, MS_PER_SECOND, MS_PER_MINUTE } from '@/lib/format-bytes'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'
import type { TrainingJob } from '@/lib/training-controller'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

export function JobHistoryCard({
  allJobs,
  checkpoints,
  loadingTimedOut,
  onRetry,
}: {
  allJobs: TrainingJob[]
  checkpoints: UseTrainingCheckpointsReturn
  loadingTimedOut: boolean
  onRetry: () => void
}) {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const [showComparison, setShowComparison] = useState(false)
  const [loadingAction, setLoadingAction] = useState<string | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [batchDeleting, setBatchDeleting] = useState(false)

  const completedJobs = allJobs.filter(j => j.status === 'completed')
  const totalLoss = completedJobs.reduce((sum, j) => sum + (j.loss ?? 0), 0)
  const avgLoss = completedJobs.length > 0 ? totalLoss / completedJobs.length : 0
  const totalEpochs = completedJobs.reduce((sum, j) => sum + (j.current_epoch ?? 0), 0)

  const overlaidLossData = useMemo(() => {
    if (completedJobs.length < 2) return null
    const jobsWithHistory = completedJobs.filter(j => j.loss_history && j.loss_history.length > 1)
    if (jobsWithHistory.length < 2) return null
    const maxLen = Math.max(...jobsWithHistory.map(j => j.loss_history?.length ?? 0))
    const data: Array<Record<string, number>> = []
    for (let i = 0; i < maxLen; i++) {
      const point: Record<string, number> = { step: i + 1 }
      jobsWithHistory.forEach(j => {
        const entry = j.loss_history?.[i]
        if (entry) point[j.name || j.id] = entry.value
      })
      data.push(point)
    }
    const keys = jobsWithHistory.map(j => j.name || j.id)
    return { data, keys }
  }, [completedJobs])

  const exportComparison = () => {
    const data = completedJobs.map(j => ({
      name: j.name || j.id,
      model: j.model,
      dataset: j.dataset,
      loss: j.loss,
      epochs: j.current_epoch,
      method: j.method,
      created_at: j.created_at,
    }))
    downloadJson(data, `training-comparison-${todayDateString()}.json`)
    addToast(`Exported ${data.length} jobs`, 'success')
  }

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const toggleSelectAll = () => {
    if (selectedIds.size === allJobs.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(allJobs.map(j => j.id)))
    }
  }

  const handleBatchDelete = async () => {
    if (selectedIds.size === 0) return
    setBatchDeleting(true)
    try {
      await Promise.all(Array.from(selectedIds).map(id => trainingController.delete(id)))
      setSelectedIds(new Set())
      addToast(`Deleted ${selectedIds.size} jobs`, 'success')
      void checkpoints.fetchJobs()
    } catch {
      addToast('Batch delete failed', 'error')
    } finally {
      setBatchDeleting(false)
    }
  }

  const hasJobs = allJobs.length > 0 || checkpoints.loadingJobs
  if (allJobs.length === 0 && !checkpoints.loadingJobs) return (
    <Card>
      <CardHeader><CardTitle className="text-base">Training Jobs</CardTitle></CardHeader>
      <CardContent>
        <p className="text-xs text-muted-foreground text-center py-4">No training jobs yet. Start training to see history here.</p>
      </CardContent>
    </Card>
  )

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Job history</CardTitle>
          {completedJobs.length > 1 && (
            <div className="flex items-center gap-1">
              <Button size="sm" variant={showComparison ? 'default' : 'ghost'} className="h-6 text-xs" onClick={() => setShowComparison(!showComparison)}>
                {showComparison ? 'List' : 'Compare'}
              </Button>
              {showComparison && (
                <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={exportComparison}>
                  Export
                </Button>
              )}
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {completedJobs.length > 0 && (
          <div className="mb-4">
            <KpiGrid columns={4}>
              <StatCard label="Total jobs" value={allJobs.length} />
              <StatCard label="Completed" value={completedJobs.length} />
              <StatCard label="Avg loss" value={avgLoss > 0 ? avgLoss.toFixed(4) : '—'} />
              <StatCard label="Total epochs" value={totalEpochs} />
            </KpiGrid>
          </div>
        )}
        <div className="p-0">
        {showComparison && completedJobs.length > 0 ? (
          <>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border/50 text-muted-foreground">
                  <th className="text-left px-4 py-2 font-medium">Name</th>
                  <th className="text-left px-4 py-2 font-medium">Model</th>
                  <th className="text-left px-4 py-2 font-medium">Dataset</th>
                  <th className="text-right px-4 py-2 font-medium">Loss</th>
                  <th className="text-right px-4 py-2 font-medium">Epochs</th>
                  <th className="text-left px-4 py-2 font-medium">Method</th>
                </tr>
              </thead>
              <tbody>
                {completedJobs.map(job => {
                  const bestLoss = Math.min(...completedJobs.map(j => j.loss ?? Infinity))
                  const worstLoss = Math.max(...completedJobs.map(j => j.loss ?? 0))
                  const loss = job.loss ?? 0
                  return (
                    <tr key={job.id} className="border-b border-border/30 hover:bg-muted/20 cursor-pointer" onClick={() => router.push(`/training/job/${job.id}`)}>
                      <td className="px-4 py-2 font-medium truncate max-w-[150px]">{job.name || job.id}</td>
                      <td className="px-4 py-2 text-muted-foreground">{job.model || '—'}</td>
                      <td className="px-4 py-2 text-muted-foreground">{job.dataset || '—'}</td>
                      <td className={`px-4 py-2 text-right font-mono ${loss === bestLoss && loss > 0 ? 'text-success' : loss === worstLoss ? 'text-warning' : ''}`}>
                        {loss > 0 ? loss.toFixed(4) : '—'}
                      </td>
                      <td className="px-4 py-2 text-right font-mono">{job.current_epoch ?? '—'}</td>
                      <td className="px-4 py-2 text-muted-foreground">{job.method || '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {overlaidLossData && (
            <div className="px-4 py-3 border-t border-border/30">
              <p className="text-xs text-muted-foreground mb-2">Loss curves overlay</p>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={overlaidLossData.data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="step" tick={{ fontSize: 10 }} stroke="var(--muted-foreground)" />
                    <YAxis tick={{ fontSize: 10 }} stroke="var(--muted-foreground)" />
                    <Tooltip contentStyle={{ fontSize: 11, borderRadius: 6, border: '1px solid var(--border)' }} labelFormatter={(label) => `Step ${label}`} formatter={(value: number) => [value.toFixed(4), 'Loss']} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    {overlaidLossData.keys.map((key, i) => (
                      <Line
                        key={key}
                        type="monotone"
                        dataKey={key}
                        stroke={['var(--primary)', 'var(--warning)', 'var(--accent)', 'var(--success)', 'var(--destructive)'][i % 5]}
                        dot={false}
                        strokeWidth={1.5}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
          </>
        ) : loadingTimedOut ? (
            <div className="px-4 py-6 text-center space-y-2">
              <p className="text-sm text-muted-foreground">Taking longer than expected</p>
              <Button size="sm" variant="ghost" onClick={onRetry}>
                Retry
              </Button>
            </div>
        ) : checkpoints.loadingJobs ? (
            <div className="divide-y divide-border/50">
              {[1,2,3].map(i => (
                <div key={i} className="flex items-center justify-between px-4 py-3">
                  <div className="space-y-1.5 flex-1">
                    <Skeleton className="h-4 w-48" />
                    <Skeleton className="h-3 w-32" />
                  </div>
                  <Skeleton className="h-5 w-12 rounded-full" />
                </div>
              ))}
            </div>
        ) : (
          <>
            {allJobs.length > 2 && (
              <div className="px-4 py-2 border-b border-border/30 flex items-center gap-3">
                <label className="flex items-center gap-2 text-[10px] text-muted-foreground cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedIds.size === allJobs.length && allJobs.length > 0}
                    onChange={toggleSelectAll}
                    className="rounded border-border"
                  />
                  Select all ({allJobs.length})
                </label>
                {selectedIds.size > 0 && (
                  <div className="flex items-center gap-2 ml-auto">
                    <span className="text-[10px] text-destructive font-medium">{selectedIds.size} selected</span>
                    <Button size="sm" variant="ghost" className="text-destructive h-6 text-[10px]" onClick={handleBatchDelete} disabled={batchDeleting}>
                      {batchDeleting ? 'Deleting...' : 'Delete Selected'}
                    </Button>
                    <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => setSelectedIds(new Set())}>
                      Clear
                    </Button>
                  </div>
                )}
              </div>
            )}
            <div className="divide-y divide-border/50">
            {allJobs.slice().reverse().map((job) => {
              const relativeTime = (() => {
                if (!job.created_at) return ''
                const diff = Date.now() - new Date(job.created_at).getTime()
                const mins = Math.floor(diff / MS_PER_MINUTE)
                if (mins < 1) return 'just now'
                if (mins < 60) return `${mins}m ago`
                const hrs = Math.floor(mins / 60)
                if (hrs < 24) return `${hrs}h ago`
                return `${Math.floor(hrs / 24)}d ago`
              })()
              const elapsed = (() => {
                if (!job.created_at) return ''
                const end = job.finished_at ? new Date(job.finished_at).getTime() : Date.now()
                const diff = end - new Date(job.created_at).getTime()
                const secs = Math.floor(diff / 1000)
                if (secs < 60) return `${secs}s`
                const mins = Math.floor(secs / 60)
                return mins < 60 ? `${mins}m` : `${Math.floor(mins / 60)}h ${mins % 60}m`
              })()
              const isSelected = selectedIds.has(job.id)
              return (
              <div key={job.id} role="button" tabIndex={0} className={`flex items-center justify-between px-4 py-3 text-sm cursor-pointer hover:bg-muted/20 transition-colors ${isSelected ? 'bg-primary/5' : ''}`} onClick={() => router.push(`/training/job/${job.id}`)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); router.push(`/training/job/${job.id}`) } }} aria-label={`View job ${job.name || job.id}`}>
                <div className="flex items-start gap-3 min-w-0 flex-1">
                  {allJobs.length > 2 && (
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => toggleSelect(job.id)}
                      onClick={e => e.stopPropagation()}
                      className="mt-1 rounded border-border shrink-0"
                    />
                  )}
                  <div className="min-w-0 flex-1">
                  <p className="truncate font-medium">{job.name || job.id}</p>
                  <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                    {job.model && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">{job.model}</span>
                    )}
                    {job.dataset && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-secondary text-secondary-foreground font-medium">{job.dataset}</span>
                    )}
                    {job.loss != null && (
                      <span className="text-[10px] text-muted-foreground">loss {job.loss.toFixed(4)}</span>
                    )}
                    {job.current_epoch != null && job.epochs != null && (
                      <span className="text-[10px] text-muted-foreground">epoch {job.current_epoch}/{job.epochs}</span>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {job.status_message || (
                      <>{job.status} · {relativeTime}{elapsed && <> · {elapsed}</>}</>
                    )}
                    {job.status === 'running' && job.progress > 0 && job.created_at && (() => {
                      const elapsedMs = Date.now() - new Date(job.created_at).getTime()
                      const rate = job.progress / elapsedMs
                      const remainingMs = rate > 0 ? (100 - job.progress) / rate : 0
                      const mins = Math.floor(remainingMs / MS_PER_MINUTE)
                      const secs = Math.floor((remainingMs % MS_PER_MINUTE) / MS_PER_SECOND)
                      return <span className="text-warning"> · ETA {mins > 0 ? `${mins}m ${secs}s` : `${secs}s`}</span>
                    })()}
                  </p>
                </div>
                <div className="flex items-center gap-2 shrink-0 ml-3" onClick={e => e.stopPropagation()}>
                  {job.status === 'running' && (
                    <>
                      <span className="relative flex h-2 w-2"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success/60" /><span className="relative inline-flex h-2 w-2 rounded-full bg-success" /></span>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 text-xs text-destructive hover:text-destructive"
                        disabled={loadingAction === `stop-${job.id}`}
                        onClick={async () => {
                          setLoadingAction(`stop-${job.id}`)
                          try { await trainingController.stop(job.id); addToast('Stopped', 'info'); void checkpoints.fetchJobs() }
                          catch { addToast('Failed to stop job', 'error') }
                          finally { setLoadingAction(null) }
                        }}
                      >
                        {loadingAction === `stop-${job.id}` ? '...' : 'Stop'}
                      </Button>
                    </>
                  )}
                  {job.status === 'completed' && (
                    <>
                      {job.checkpoint && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-6 text-xs"
                          disabled={loadingAction === `use-${job.id}`}
                          onClick={async () => {
                            setLoadingAction(`use-${job.id}`)
                            try { const cp = job.checkpoint; if (cp) await checkpoints.handleLoadCheckpoint(cp, addToast) }
                            catch { addToast('Failed to load trained version', 'error') }
                            finally { setLoadingAction(null) }
                          }}
                        >
                          {loadingAction === `use-${job.id}` ? '...' : 'Use'}
                        </Button>
                      )}
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-success/15 text-success font-medium shrink-0">Done</span>
                    </>
                  )}
                  {job.status === 'failed' && (
                    <>
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-destructive/15 text-destructive font-medium shrink-0">Failed</span>
                      <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => router.push('/training')}>
                        Retry
                      </Button>
                    </>
                  )}
                  {['stopping', 'stopped'].includes(job.status) && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium shrink-0">Stopped</span>
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 text-xs text-muted-foreground hover:text-destructive"
                    disabled={loadingAction === `delete-${job.id}`}
                    onClick={async () => {
                      if (!confirm(`Delete job "${job.name || job.id}"?`)) return
                      setLoadingAction(`delete-${job.id}`)
                      try {
                        await trainingController.delete(job.id)
                        addToast('Job deleted', 'info')
                        void checkpoints.fetchJobs()
                      } catch { addToast('Failed to delete job', 'error') }
                      finally { setLoadingAction(null) }
                    }}
                  >
                    {loadingAction === `delete-${job.id}` ? '...' : 'Delete'}
                  </Button>
                </div>
                </div>
              </div>
              )
            })}
          </div>
          </>
        )}
      </div>
      </CardContent>
    </Card>
  )
}
