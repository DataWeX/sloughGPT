'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, cn } from '@sloughgpt/strui'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { Breadcrumbs } from '@sloughgpt/strui'
import dynamicNext from 'next/dynamic'
import type { LossPoint, RewardPoint } from '@/components/training/LossChart'

const LossChart = dynamicNext(() => import('@/components/training/LossChart').then(m => m.LossChart), { ssr: false })
import { IconTrash, IconRefresh, IconDownload } from '@sloughgpt/strui'
import { trainingJobsController, type TrainingJob } from '@/lib/training-controller'
import { modelController } from '@/lib/model-controller'
import { useToastStore } from '@/lib/toast-store'
import { downloadBlob, downloadJson } from '@/lib/download-utils'
import { formatElapsed } from '@/lib/formatDuration'

const STATUS_BADGE: Record<string, { label: string; variant: 'default' | 'secondary' | 'destructive' | 'outline' | 'success' | 'warning' | 'error' }> = {
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
  const [showDelete, setShowDelete] = useState(false)

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
          addToast('Lost connection to training service', 'error')
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
    } catch { addToast('Could not load trained version', 'error') }
  }

  const handleDelete = async () => {
    if (!job) return
    try {
      await trainingJobsController.delete(job.id)
      addToast('Job deleted', 'info')
      router.push('/training')
    } catch {       addToast('Something went wrong deleting the job', 'error')
    } finally { setShowDelete(false) }
  }

  const handleExport = () => {
    if (!job) return
    const data = {
      id: job.id,
      name: job.name,
      status: job.status,
      model: job.model,
      dataset: job.dataset,
      epochs: job.epochs,
      current_epoch: job.current_epoch,
      loss: job.loss,
      checkpoint: job.checkpoint,
      created_at: job.created_at,
      finished_at: job.finished_at,
      error: job.error,
      loss_history: job.loss_history,
    }
    downloadJson(data, `training-job-${job.id}.json`)
    addToast('Job details exported', 'success')
  }

  const headerRight = (
    <div className="flex items-center gap-1">
      <Button variant="ghost" size="sm" onClick={fetchJob} disabled={loading} aria-label="Refresh job status">
        <IconRefresh className={loading ? 'animate-spin h-4 w-4' : 'h-4 w-4'} />
      </Button>
      <Button variant="ghost" size="sm" onClick={handleExport} disabled={!job} aria-label="Export job details">
        <IconDownload className="h-4 w-4" />
      </Button>
      <Button variant="ghost" size="sm" className="text-destructive" onClick={() => setShowDelete(true)} disabled={!job} aria-label="Delete job">
        <IconTrash className="h-4 w-4" />
      </Button>
    </div>
  )

  return (
    <PageContainer
      title={loading ? '...' : job?.name || jobId}
      headerRight={headerRight}
      loading={loading}
      loadingContent={
        <div className="space-y-3">
          <Skeleton className="h-32 rounded-lg" />
          <Skeleton className="h-48 rounded-lg" />
        </div>
      }
    >
      <Breadcrumbs
        items={[
          { label: 'Training', href: '/training' },
          { label: loading ? '...' : job?.name || jobId },
        ]}
        className="mb-3"
      />

      {!job ? (
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
                  <Skeleton className="h-4 w-4/4" />
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
              <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2">
                  <Badge variant={badge?.variant ?? 'outline'}>{badge?.label}</Badge>
                  {job.status === 'running' && (
                    <span className="relative flex h-2 w-2">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
                    </span>
                  )}
                </div>
                <div className="flex flex-wrap items-center gap-1">
                  {job.status === 'running' && (
                    <Button size="sm" variant="outline" className="h-8 text-xs text-destructive border-destructive/30 hover:bg-destructive/10" onClick={async () => {
                      try {
                        await trainingJobsController.stop(job.id)
                        addToast('Training stopped', 'info')
                        await fetchJob()
                      } catch { addToast('Could not stop training', 'error') }
                    }}>
                      Stop
                    </Button>
                  )}
                  {job.checkpoint && job.status === 'completed' && (
                    <>
                      <Button size="sm" variant="outline" className="h-8 text-xs" onClick={handleLoadCheckpoint}>
                        Load saved version
                      </Button>
                      <Button size="sm" variant="ghost" className="h-8 text-xs" onClick={async () => {
                        try {
                          const blob = await trainingJobsController.downloadTrainingJob(job.id)
                          downloadBlob(blob, `${job.id}.checkpoint`)
                          addToast('Checkpoint downloaded', 'success')
                        } catch { addToast('Could not download', 'error') }
                      }}>
                        <IconDownload className="h-4 w-4 mr-1" /> Export
                      </Button>
                    </>
                  )}
                  <Button size="sm" variant="outline" className="h-8 text-xs" onClick={async () => {
                    if (job?.checkpoint) {
                      try {
                        await modelController.loadModelPath(job.checkpoint)
                        addToast(`Loaded trained version: ${job.checkpoint}`, 'success')
                      } catch { addToast('Could not load model', 'error') }
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
                  <div className="flex items-center justify-between mt-1">
                    <p className="text-xs text-muted-foreground">{job.progress}%</p>
                    {job.progress > 0 && job.created_at && (() => {
                      const elapsed = (Date.now() - new Date(job.created_at).getTime()) / 1000
                      const rate = job.progress / elapsed
                      const remaining = rate > 0 ? (100 - job.progress) / rate : 0
                      const mins = Math.floor(remaining / 60)
                      const secs = Math.floor(remaining % 60)
                      return (
                        <p className="text-xs text-muted-foreground">
                          ETA: {mins > 0 ? `${mins}m ${secs}s` : `${secs}s`}
                        </p>
                      )
                    })()}
                  </div>
                </div>
              )}

              <KpiGrid columns={4}>
                <StatCard label="Model" value={job.model || '—'} />
                <StatCard label="Dataset" value={job.dataset || '—'} />
                <StatCard label="Epochs" value={job.epochs != null ? `${job.current_epoch ?? 0} / ${job.epochs}` : '—'} />
                <StatCard label="Steps" value={job.global_step != null ? String(job.global_step) : '—'} />
              </KpiGrid>
              {job.status === 'running' && job.global_step && job.created_at && (() => {
                const elapsed = (Date.now() - new Date(job.created_at).getTime()) / 1000
                const stepsPerMin = elapsed > 0 ? (job.global_step / elapsed) * 60 : 0
                return stepsPerMin > 0 ? (
                  <div className="mt-2 flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">Speed:</span>
                    <span className="text-xs font-mono text-primary">{stepsPerMin.toFixed(1)} steps/min</span>
                  </div>
                ) : null
              })()}
            </CardContent>
          </Card>

          {/* Loss card */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Loss &amp; Reward</CardTitle>
                {job.loss_history && job.loss_history.length >= 3 && (() => {
                  const recent = job.loss_history.slice(-5)
                  const firstHalf = recent.slice(0, Math.floor(recent.length / 2))
                  const secondHalf = recent.slice(Math.floor(recent.length / 2))
                  const avgFirst = firstHalf.reduce((s, p) => s + p.value, 0) / firstHalf.length
                  const avgSecond = secondHalf.reduce((s, p) => s + p.value, 0) / secondHalf.length
                  const improving = avgSecond < avgFirst
                  const pctChange = avgFirst > 0 ? ((avgSecond - avgFirst) / avgFirst * 100).toFixed(1) : '0'
                  return (
                    <span className={cn('text-xs font-medium px-2 py-0.5 rounded', improving ? 'bg-success/10 text-success' : 'bg-warning/10 text-warning')}>
                      {improving ? '↓' : '↑'} {Math.abs(Number(pctChange))}%
                    </span>
                  )
                })()}
              </div>
            </CardHeader>
            <CardContent>
              <KpiGrid columns={job.reward_history?.length ? 4 : 3}>
                {job.loss != null && <StatCard label="Final loss" value={job.loss.toFixed(4)} />}
                {job.train_loss != null && <StatCard label="Train loss" value={job.train_loss.toFixed(4)} />}
                {job.eval_loss != null && <StatCard label="Validation loss" value={job.eval_loss.toFixed(4)} />}
                {typeof job.result?.final_reward === 'number' && <StatCard label="Final reward" value={job.result.final_reward.toFixed(4)} />}
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
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
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
                  <p className="text-xs mt-0.5">{formatElapsed(job.created_at, job.finished_at)}</p>
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

      <AlertDialog open={showDelete} onOpenChange={setShowDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete job</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &ldquo;{job?.name || job?.id}&rdquo;? This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </PageContainer>
  )
}
