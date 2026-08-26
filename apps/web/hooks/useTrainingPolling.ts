'use client'

import { useCallback, useRef, useEffect } from 'react'
import { trainingJobsController, type TrainingJob } from '@/lib/controllers'
import { readTraining, writeTraining, type TrainingToastFn } from '@/lib/app-shell'

const MAX_POLL_RETRIES = 10
const STANDARD_BASE_DELAY_MS = 3000
const STANDARD_MAX_DELAY_MS = 30000
const TURBO_BASE_DELAY_MS = 3000
const TURBO_MAX_DELAY_MS = 60000

function sendBrowserNotification(title: string, body: string) {
  if (typeof window !== 'undefined' && 'Notification' in window && Notification.permission === 'granted') {
    try {
      new Notification(title, { body, icon: '/favicon.svg' })
    } catch { /* Notification API unavailable in this context */ }
  }
}

export interface TrainingPolling {
  startStandardPoll: (jobId: string, opts?: { addToast?: TrainingToastFn; onComplete?: (job: TrainingJob) => void; completeMessage?: string }) => void
  startTurboPoll: (addToast?: TrainingToastFn) => void
  clearAllPolls: () => void
}

/**
 * Manages training poll intervals. Reads fresh shell state on every tick
 * via readTraining() — no stale closures, no React state.
 *
 * - Standard poll: GET /training/jobs/{id} every 3s.
 * - Turbo poll: GET /auto-train/status every 3s.
 *
 * Resilient: retries on transient network errors, warns user after
 * MAX_POLL_RETRIES consecutive failures, only kills poll after threshold.
 */
export function useTrainingPolling(): TrainingPolling {
  const standardPollRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const turboPollRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const standardRetryRef = useRef(0)
  const turboRetryRef = useRef(0)

  const clearAllPolls = useCallback(() => {
    if (standardPollRef.current) { clearTimeout(standardPollRef.current); standardPollRef.current = null }
    if (turboPollRef.current) { clearTimeout(turboPollRef.current); turboPollRef.current = null }
    standardRetryRef.current = 0
    turboRetryRef.current = 0
  }, [])

  const startStandardPoll = useCallback((
    jobId: string,
    opts?: { addToast?: TrainingToastFn; onComplete?: (job: TrainingJob) => void; completeMessage?: string },
  ) => {
    if (standardPollRef.current) { clearInterval(standardPollRef.current); standardPollRef.current = null }
    standardRetryRef.current = 0

    const poll = async () => {
      try {
        const job = await trainingJobsController.get(jobId)
        standardRetryRef.current = 0
        if (!job) { standardPollRef.current = null; return }

        if (job.status === 'running') {
          const current = readTraining()
          const patch: Record<string, unknown> = {
            progress: job.progress ?? current.progress,
            loss: job.loss ?? job.train_loss ?? current.loss,
            epoch: job.current_epoch ?? current.epoch,
            totalEpochs: job.epochs ?? current.totalEpochs,
            globalStep: job.global_step ?? current.globalStep,
            totalSteps: job.total_steps ?? current.totalSteps,
            stepsPerSec: job.steps_per_sec ?? current.stepsPerSec,
            eta: job.eta_s ?? current.eta,
            elapsedSeconds: job.elapsed_s ?? current.elapsedSeconds,
            avgQuality: job.avg_quality ?? current.avgQuality,
          }
          if (job.loss != null || job.train_loss != null) {
            const loss = (job.loss ?? job.train_loss) as number
            const step = job.global_step ?? current.globalStep
            const hist = [...current.lossHistory, { step: step || current.lossHistory.length, loss }]
            patch.lossHistory = hist.length > 200 ? hist.slice(-200) : hist
          }
          writeTraining(patch)
          standardPollRef.current = setTimeout(poll, STANDARD_BASE_DELAY_MS)
          return
        }

        if (job.status === 'completed') {
          const result = job.result as Record<string, unknown> | undefined
          writeTraining({
            phase: 'complete', progress: 100,
            checkpoint: job.checkpoint ?? null,
            finalLoss: (result?.final_loss as number) ?? job.loss ?? job.train_loss ?? null,
            modelPath: (result?.model_path as string) ?? null,
            avgQuality: job.avg_quality ?? (result?.avg_quality as number) ?? null,
          })
          opts?.addToast?.(opts.completeMessage ?? 'Training complete', 'success')
          sendBrowserNotification('Training Complete', `${job.name || 'Training job'} finished successfully`)
          opts?.onComplete?.(job)
        } else {
          writeTraining({ phase: 'error', error: job.error || 'Could not train' })
          opts?.addToast?.(job.error || 'Could not train', 'error')
          sendBrowserNotification('Training Failed', job.error || 'Training encountered an error')
        }
      } catch (e) {
        standardRetryRef.current++
        if (standardRetryRef.current >= MAX_POLL_RETRIES) {
          standardPollRef.current = null
          writeTraining({ phase: 'error', error: 'Lost connection to training service' })
          opts?.addToast?.('Lost connection to training — check server status', 'error')
          sendBrowserNotification('Training Lost', 'Lost connection to training service')
          return
        }
        const delay = Math.min(STANDARD_BASE_DELAY_MS * Math.pow(2, standardRetryRef.current - 1), STANDARD_MAX_DELAY_MS)
        standardPollRef.current = setTimeout(poll, delay)
      }
    }
    standardPollRef.current = setTimeout(poll, STANDARD_BASE_DELAY_MS)
  }, [])

  const startTurboPoll = useCallback((addToast?: TrainingToastFn) => {
    if (turboPollRef.current) { clearInterval(turboPollRef.current); turboPollRef.current = null }
    turboRetryRef.current = 0

    const pollId = setInterval(async () => {
      try {
        const s = await trainingJobsController.getTurboStatus()
        turboRetryRef.current = 0
        const current = readTraining()

        if (s.status === 'running') {
          const patch: Record<string, unknown> = {
            progress: s.progress ?? current.progress,
            loss: s.loss ?? current.loss,
            globalStep: s.global_step ?? current.globalStep,
            totalSteps: s.total_steps ?? current.totalSteps,
            stepsPerSec: s.steps_per_sec ?? current.stepsPerSec,
            eta: s.eta_s ?? current.eta,
            elapsedSeconds: s.elapsed_s ?? current.elapsedSeconds,
            avgQuality: s.avg_quality ?? current.avgQuality,
            message: s.paused ? 'Paused' : (current.message === 'Paused' ? '' : current.message),
          }
          if (s.loss != null) {
            const step = s.global_step ?? current.globalStep
            const hist = [...current.lossHistory, { step: step || current.lossHistory.length, loss: s.loss }]
            patch.lossHistory = hist.length > 200 ? hist.slice(-200) : hist
          }
          writeTraining(patch)
          return
        }

        if (s.status === 'error' && s.error) {
          clearInterval(pollId); turboPollRef.current = null
          writeTraining({ phase: 'error', error: s.error })
          addToast?.(s.error, 'error')
          sendBrowserNotification('Turbo Training Failed', s.error)
          return
        }

        clearInterval(pollId); turboPollRef.current = null

        if (s.status === 'idle') {
          // Turbo service went idle — training was interrupted or never started
          writeTraining({ phase: 'idle' })
          return
        }

        if (s.status === 'complete') {
          writeTraining({
            phase: 'complete', progress: 100,
            checkpoint: (s.result?.checkpoint as string) ?? null,
            finalLoss: (s.result?.final_loss as number) ?? null,
            modelPath: (s.result?.model_path as string) ?? null,
            avgQuality: (s.result?.avg_quality as number) ?? s.avg_quality ?? null,
          })
          addToast?.('Turbo training complete!', 'success')
          sendBrowserNotification('Turbo Training Complete', 'Your turbo training finished successfully')
        } else {
          writeTraining({ phase: 'error', error: s.error || 'Could not train' })
          addToast?.(s.error || 'Could not train', 'error')
          sendBrowserNotification('Turbo Training Failed', s.error || 'Turbo training encountered an error')
        }
      } catch (e) {
        turboRetryRef.current++
        if (turboRetryRef.current >= MAX_POLL_RETRIES) {
          clearInterval(pollId); turboPollRef.current = null
          writeTraining({ phase: 'error', error: 'Lost connection to training service' })
          addToast?.('Lost connection to training — check server status', 'error')
          sendBrowserNotification('Training Lost', 'Lost connection to turbo training service')
        }
      }
    }, 3000)
    turboPollRef.current = pollId
  }, [])

  useEffect(() => () => clearAllPolls(), [clearAllPolls])

  return { startStandardPoll, startTurboPoll, clearAllPolls }
}
