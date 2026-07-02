'use client'

import { useCallback, useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/display'
import { Badge } from '@/components/ui/badge'
import { StatCard, KpiGrid } from '@/components/ui'
import dynamic from 'next/dynamic'
import type { LossPoint, RewardPoint } from '@/components/training/LossChart'

const LossChart = dynamic(() => import('@/components/training/LossChart').then(m => m.LossChart), { ssr: false })
import { IconTrash, IconRefresh, IconDownload } from '@/components/ui'
import { trainingJobsController, type TrainingJob } from '@/lib/training-controller'
import { modelController } from '@/lib/model-controller'
import { useToastStore } from '@/lib/toast-store'

function formatDuration(start: number | string, end?: number | string): string {
  const s = new Date(start).getTime()
  const e = end ? new Date(end).getTime() : Date.now()
  const ms = e - s
  if (ms < 60000) return `${Math.floor(ms / 1000)}s`
  if (ms < 3600000) return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`
  return `${Math.floor(ms / 3600000)}h ${Math.floor((ms % 3600000) / 60000)}m`
}

const STATUS_BADGE: Record<string, { label: string; variant: string }> = {
  running: { label: 'Running', variant: 'default' },
  completed: { label: 'Completed', variant: 'secondary' },
  failed: { label: 'Failed', variant: 'destructive' },
  stopped: { label: 'Stopped', variant: 'secondary' },
  queued: { label: 'Queued', variant: 'outline' },
}

export default function TrainingJobDetailPage() {
  const params = useParams()
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const jobId = (params.id as string) || ''

  const [job, setJob] = useState<TrainingJob | null>(null)
  const [loading, setLoading] = useState(true)
  const [summaryText, setSummaryText] = useState<string | null>(null)
  const [summaryLoading, setSummaryLoading] = useState(false)

  const fetchJob = useCallback(async () => {
    if (!jobId) return
    setLoading(true)
    try {
      const j = await trainingJobsController.get(jobId)
      setJob(j)
    } catch {
      addToast('Something went wrong loading the job', 'error')
    } finally {
      setLoading(false)
    }
  }, [jobId, addToast])

  const fetchSummary = useCallback(async () => {
    if (!jobId) return
    setSummaryLoading(true)
    try {
      const res = await trainingJobsController.getSummary(jobId)
      setSummaryText(res.summary)
    } catch {
      // summary is optional — no toast
    } finally {
      setSummaryLoading(false)
    }
  }, [jobId])

  useEffect(() => {
    if (!jobId) { router.push('/training'); return }
    void fetchJob()
    void fetchSummary()
  }, [jobId, fetchJob, fetchSummary, router])

  // Poll if running
  useEffect(() => {
    if (job?.status !== 'running') return
    let consecutiveErrors = 0
    const id = setInterval(async () => {
      try {
        await fetchJob()
        consecutiveErrors = 0
      } catch {
        consecutiveErrors++
        if (consecutiveErrors >= 5) {
          clearInterval(id)
          addToast('Lost connection to training server', 'error')
        }
      }
    }, 3000)
    return () => clearInterval(id)
  }, [job?.status, fetchJob, addToast])

  const badge = job ? STATUS_BADGE[job.status] || { label: job.status, variant: 'outline' } : null

  const handleLoadCheckpoint = async () => {
    if (!job?.checkpoint) return
    try {
      await modelController.loadModelPath(job.checkpoint)
      addToast(`Loaded trained version: ${job.checkpoint}`, 'success')
    } catch { addToast('Failed to load trained version', 'error') }
  }

  const handleDelete = async () => {
    if (!job) return
    if (!confirm(`Delete job "${job.name || job.id}"?`)) return
    try {
      await trainingJobsController.delete(job.id)
      addToast('Job deleted', 'info')
      router.push('/training')
    } catch {       addToast('Something went wrong deleting the job', 'error') }
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm" onClick={() => router.push('/training')} className="h-7 px-1.5 text-xs text-muted-foreground hover:text-foreground">
              ← Training
            </Button>
            <AppRouteHeaderLead title={loading ? '...' : job?.name || jobId} />
          </div>
        }
        right={
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="sm" onClick={fetchJob} disabled={loading} aria-label="Refresh job status">
              <IconRefresh className={loading ? 'animate-spin h-3.5 w-3.5' : 'h-3.5 w-3.5'} />
            </Button>
            <Button variant="ghost" size="sm" className="text-destructive" onClick={handleDelete} disabled={!job} aria-label="Delete job">
              <IconTrash className="h-3.5 w-3.5" />
            </Button>
          </div>
        }
      />

      <div className="space-y-4">
        {loading ? (
          <div className="space-y-3">
            <Skeleton className="h-32 rounded-lg" />
            <Skeleton className="h-48 rounded-lg" />
          </div>
        ) : !job ? (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">Job not found</CardContent>
          </Card>
        ) : (
          <>
            {/* Plain-language explanation when completed */}
            {job.status === 'completed' && job.explanation && (
              <div className="rounded-lg border border-success/20 bg-success/5 p-4">
                <p className="text-sm font-medium text-success">{job.explanation}</p>
              </div>
            )}

            {/* Summary card */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Summary</CardTitle>
              </CardHeader>
              <CardContent>
                {summaryLoading ? (
                  <div className="space-y-2">
                    <Skeleton className="h-3 w-full" />
                    <Skeleton className="h-3 w-3/4" />
                    <Skeleton className="h-3 w-1/2" />
                  </div>
                ) : summaryText ? (
                  <div className="text-sm text-muted-foreground whitespace-pre-wrap leading-relaxed">{summaryText}</div>
                ) : (
                  <p className="text-sm text-muted-foreground">Summary not available</p>
                )}
              </CardContent>
            </Card>

            {/* Status + actions */}
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Badge variant={badge?.variant as any}>{badge?.label}</Badge>
                    {job.status === 'running' && (
                      <span className="relative flex h-2 w-2">
                        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
                        <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    {job.checkpoint && job.status === 'completed' && (
                      <>
                        <Button size="sm" variant="outline" className="h-7 text-xs" onClick={handleLoadCheckpoint}>
                          Load saved version
                        </Button>
                        <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={async () => {
                          try {
                            const blob = await trainingJobsController.downloadTrainingJob(job.id)
                            const url = URL.createObjectURL(blob)
                            const a = document.createElement('a')
                            a.href = url; a.download = `${job.id}.checkpoint`
                            a.click(); URL.revokeObjectURL(url)
                            addToast('Checkpoint downloaded', 'success')
                          } catch { addToast('Download failed', 'error') }
                        }}>
                          <IconDownload className="h-3 w-3 mr-1" /> Export
                        </Button>
                      </>
                    )}
                    <Button size="sm" variant="outline" className="h-7 text-xs" onClick={async () => {
                      if (job?.checkpoint) {
                        try {
                          await modelController.loadModelPath(job.checkpoint)
                          addToast(`Loaded trained version: ${job.checkpoint}`, 'success')
                        } catch { addToast('Failed to load model', 'error') }
                      }
                      router.push('/chat')
                    }}>
                      Try in chat
                    </Button>
                  </div>
                </div>

                {/* Progress bar for running jobs */}
                {job.status === 'running' && (
                  <div className="mb-3">
                    <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-primary transition-all duration-500 rounded-full" style={{ width: `${Math.min(job.progress, 100)}%` }} />
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">{job.progress}%</p>
                  </div>
                )}

                <KpiGrid columns={4}>
                  <StatCard label="Model" value={job.model || '—'} />
                  <StatCard label="Dataset" value={job.dataset || '—'} />
                  <StatCard label="Epochs" value={job.epochs != null ? `${job.current_epoch ?? 0} / ${job.epochs}` : '—'} />
                  <StatCard label="Steps" value={job.global_step != null ? String(job.global_step) : '—'} />
                </KpiGrid>
              </CardContent>
            </Card>

            {/* Loss card */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Loss &amp; Reward</CardTitle>
              </CardHeader>
              <CardContent>
                <KpiGrid columns={job.reward_history?.length ? 4 : 3}>
                  {job.loss != null && <StatCard label="Final loss" value={job.loss.toFixed(4)} />}
                  {job.train_loss != null && <StatCard label="Train loss" value={job.train_loss.toFixed(4)} />}
                  {job.eval_loss != null && <StatCard label="Validation loss" value={job.eval_loss.toFixed(4)} />}
                  {(job.result as any)?.final_reward != null && <StatCard label="Final reward" value={(job.result as any).final_reward.toFixed(4)} />}
                </KpiGrid>
                {job.loss_history && job.loss_history.length > 1 && (
                  <div className="mt-4">
                    <LossChart
                      data={job.loss_history.map(p => ({ step: p.step, value: p.value, type: p.type } as LossPoint))}
                      rewardData={job.reward_history?.map(p => ({ step: p.step, value: p.value } as RewardPoint))}
                      live={job.status === 'running'}
                    />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Details card */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-xs text-muted-foreground">Job ID</p>
                    <p className="font-mono text-xs mt-0.5">{job.id}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Source</p>
                    <p className="text-xs mt-0.5">{job.data_source || '—'}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Created</p>
                    <p className="text-xs mt-0.5">{new Date(job.created_at).toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Duration</p>
                    <p className="text-xs mt-0.5">{formatDuration(job.created_at, job.finished_at)}</p>
                  </div>
                  {job.checkpoint && (
                    <div className="col-span-2">
                      <p className="text-xs text-muted-foreground">Saved version</p>
                      <p className="font-mono text-xs mt-0.5 truncate">{job.checkpoint}</p>
                    </div>
                  )}
                  {job.message && (
                    <div className="col-span-2">
                      <p className="text-xs text-muted-foreground">Message</p>
                      <p className="text-xs mt-0.5 text-muted-foreground">{job.message}</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
