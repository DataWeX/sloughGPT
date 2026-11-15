'use client'
import { notify } from '@/lib/notify'

import { useCallback, useEffect, useState } from 'react'
import { useParams, useRouter } from '@/vite/next-compat/navigation'
import { trainingJobsController, type TrainingJob } from '@/lib/training-controller'
import { modelController } from '@/lib/model-controller'
import { downloadBlob, downloadJson } from '@/lib/download-utils'
import { formatToastError } from '@/lib/error-utils'
// Module scope: `addToast` must keep a stable identity — it sits in
// useCallback/useEffect dependency arrays, and a per-render arrow would
// re-fire those effects on every render.
const addToast = (message: string, type?: string) =>
  notify.toast(message, { type: type as 'success' | 'error' | 'info' })

export interface JobBadge {
  label: string
  variant: 'default' | 'secondary' | 'destructive' | 'outline' | 'success' | 'warning' | 'error'
}

const STATUS_BADGE: Record<string, JobBadge> = {
  running: { label: 'Running', variant: 'default' },
  completed: { label: 'Completed', variant: 'secondary' },
  failed: { label: 'Failed', variant: 'destructive' },
  interrupted: { label: 'Interrupted', variant: 'warning' },
  stopped: { label: 'Stopped', variant: 'secondary' },
  queued: { label: 'Queued', variant: 'outline' },
}

export interface UseTrainingJobReturn {
  jobId: string
  job: TrainingJob | null
  loading: boolean
  fetchError: string | null
  summaryText: string | null
  summaryLoading: boolean
  showDelete: boolean
  setShowDelete: (v: boolean) => void
  badge: JobBadge | null
  fetchJob: () => Promise<void>
  handleExport: () => void
  handleLoadCheckpoint: () => Promise<void>
  handleDelete: () => Promise<void>
  handleResume: () => Promise<void>
  handleStop: () => Promise<void>
  handleDownloadCheckpoint: () => Promise<void>
  handleTryInChat: () => Promise<void>
}

export function useTrainingJob(): UseTrainingJobReturn {
  const params = useParams()
  const router = useRouter()
  const jobId = (params.id as string) || ''

  const [job, setJob] = useState<TrainingJob | null>(null)
  const [loading, setLoading] = useState(true)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const [summaryText, setSummaryText] = useState<string | null>(null)
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [showDelete, setShowDelete] = useState(false)

  const fetchJob = useCallback(async () => {
    if (!jobId) return
    setLoading(true)
    setFetchError(null)
    try {
      const j = await trainingJobsController.get(jobId)
      setJob(j)
    } catch (e) {
      setFetchError(formatToastError(e, 'Failed to load training job'))
    } finally {
      setLoading(false)
    }
  }, [jobId])

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
    if (!jobId) {
      router.push('/training')
      return
    }
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
          notify.toast('Lost connection to training service', { type: 'error' })
        }
      }
    }, 3000)
    return () => clearInterval(id)
  }, [job?.status, fetchJob])

  const badge = job ? STATUS_BADGE[job.status] || { label: job.status, variant: 'outline' } : null

  const handleLoadCheckpoint = async () => {
    if (!job?.checkpoint) return
    try {
      await modelController.loadModelPath(job.checkpoint)
      notify.toast(`Loaded trained version: ${job.checkpoint}`, { type: 'success' })
    } catch (e) {
      notify.toast(formatToastError(e, 'Could not load trained version'), { type: 'error' })
    }
  }

  const handleDelete = async () => {
    if (!job) return
    try {
      await trainingJobsController.delete(job.id)
      notify.toast('Job deleted', { type: 'info' })
      router.push('/training')
    } catch (e) {
      notify.toast(formatToastError(e, 'Something went wrong deleting the job'), { type: 'error' })
    } finally {
      setShowDelete(false)
    }
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
    notify.toast('Job details exported', { type: 'success' })
  }

  const handleResume = async () => {
    if (!job) return
    try {
      const result = await trainingJobsController.recover(job.id)
      notify.toast(result.message || 'Resume started', { type: 'success' })
      if (result.recovery_job_id && result.recovery_job_id !== job.id) {
        router.push(`/training/job/${result.recovery_job_id}`)
      } else {
        await fetchJob()
      }
    } catch (e) {
      notify.toast(formatToastError(e, 'Could not resume job'), { type: 'error' })
    }
  }

  const handleStop = async () => {
    if (!job) return
    try {
      await trainingJobsController.stop(job.id)
      notify.toast('Training stopped', { type: 'info' })
      await fetchJob()
    } catch (e) {
      notify.toast(formatToastError(e, 'Could not stop training'), { type: 'error' })
    }
  }

  const handleDownloadCheckpoint = async () => {
    if (!job) return
    try {
      const blob = await trainingJobsController.downloadTrainingJob(job.id)
      downloadBlob(blob, `${job.id}.checkpoint`)
      notify.toast('Checkpoint downloaded', { type: 'success' })
    } catch (e) {
      notify.toast(formatToastError(e, 'Could not download'), { type: 'error' })
    }
  }

  const handleTryInChat = async () => {
    if (job?.checkpoint) {
      try {
        await modelController.loadModelPath(job.checkpoint)
        notify.toast(`Loaded trained version: ${job.checkpoint}`, { type: 'success' })
      } catch (e) {
        notify.toast(formatToastError(e, 'Could not load model'), { type: 'error' })
      }
    }
    router.push('/chat')
  }

  return {
    jobId,
    job,
    loading,
    fetchError,
    summaryText,
    summaryLoading,
    showDelete,
    setShowDelete,
    badge,
    fetchJob,
    handleExport,
    handleLoadCheckpoint,
    handleDelete,
    handleResume,
    handleStop,
    handleDownloadCheckpoint,
    handleTryInChat,
  }
}
