'use client'

import { useCallback, useRef, useEffect } from 'react'
import { trainingJobsController, type TrainingJob } from '@/lib/controllers'
import { appShellStore, readTraining, writeTraining, type TrainingShellState, type TrainingToastFn } from '@/lib/app-shell'

export interface TrainingPolling {
  startStandardPoll: (jobId: string, opts?: { addToast?: TrainingToastFn; onComplete?: (job: TrainingJob) => void; completeMessage?: string }) => void
  startTurboPoll: (addToast?: TrainingToastFn) => void
  clearAllPolls: () => void
}

/**
 * Manages training poll intervals. Pure logic — no React state, no reconciliation.
 *
 * - Standard poll: GET /training/jobs/{id} every 3s. Handles running/completed/failed.
 * - Turbo poll: GET /auto-train/status every 3s. Handles running/complete/error.
 *
 * Both read fresh state from appShellStore on each tick (no stale closures).
 */
export function useTrainingPolling(): TrainingPolling {
  const standardPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const turboPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const clearAllPolls = useCallback(() => {
    if (standardPollRef.current) { clearInterval(standardPollRef.current); standardPollRef.current = null }
    if (turboPollRef.current) { clearInterval(turboPollRef.current); turboPollRef.current = null }
  }, [])

  const startStandardPoll = useCallback((
    jobId: string,
    opts?: { addToast?: ToastFn; onComplete?: (job: TrainingJob) => void; completeMessage?: string },
  ) => {
    if (standardPollRef.current) { clearInterval(standardPollRef.current); standardPollRef.current = null }
    const pollId = setInterval(async () => {
      try {
        const job = await trainingJobsController.get(jobId)
        if (!job) { clearInterval(pollId); standardPollRef.current = null; return }
        const current = readTraining()
        if (job.status === 'running') {
          const patch: Partial<TrainingShellState> = {
            progress: job.progress ?? current.progress,
            loss: job.loss ?? job.train_loss ?? current.loss,
            epoch: job.current_epoch ?? current.epoch,
            totalEpochs: job.epochs ?? current.totalEpochs,
            globalStep: job.global_step ?? current.globalStep,
            totalSteps: job.total_steps ?? current.totalSteps,
            stepsPerSec: job.steps_per_sec ?? current.stepsPerSec,
            eta: job.eta_s ?? current.eta,
            elapsedSeconds: job.elapsed_s ?? current.elapsedSeconds,
          }
          if (job.loss != null || job.train_loss != null) {
            const loss = (job.loss ?? job.train_loss) as number
            const step = job.global_step ?? current.globalStep
            const hist = [...current.lossHistory, { step: step || current.lossHistory.length, loss }]
            patch.lossHistory = hist.length > 200 ? hist.slice(-200) : hist
          }
          writeTraining(patch)
          return
        }
        clearInterval(pollId); standardPollRef.current = null
        if (job.status === 'completed') {
          const result = job.result as Record<string, unknown> | undefined
          writeTraining({
            phase: 'complete',
            progress: 100,
            checkpoint: job.checkpoint ?? null,
            finalLoss: (result?.final_loss as number) ?? job.loss ?? job.train_loss ?? null,
            modelPath: (result?.model_path as string) ?? null,
          })
          opts?.addToast?.(opts.completeMessage ?? 'Training complete', 'success')
          opts?.onComplete?.(job)
        } else {
          const errMsg = job.error || 'Training failed'
          writeTraining({ phase: 'error', error: errMsg })
          opts?.addToast?.(errMsg, 'error')
        }
      } catch {
        clearInterval(pollId); standardPollRef.current = null
      }
    }, 3000)
    standardPollRef.current = pollId
  }, [])

  const startTurboPoll = useCallback((addToast?: ToastFn) => {
    if (turboPollRef.current) { clearInterval(turboPollRef.current); turboPollRef.current = null }
    const pollId = setInterval(async () => {
      try {
        const s = await trainingJobsController.getTurboStatus()
        const current = readTraining()
        if (s.status === 'running' || s.status === 'idle') {
          writeTraining({
            progress: s.progress ?? current.progress,
            loss: s.loss ?? current.loss,
            globalStep: s.global_step ?? current.globalStep,
            totalSteps: s.total_steps ?? current.totalSteps,
            stepsPerSec: s.steps_per_sec ?? current.stepsPerSec,
            eta: s.eta_s ?? current.eta,
            elapsedSeconds: s.elapsed_s ?? current.elapsedSeconds,
          })
          return
        }
        clearInterval(pollId); turboPollRef.current = null
        if (s.status === 'complete') {
          writeTraining({
            phase: 'complete',
            progress: 100,
            checkpoint: (s.result?.checkpoint as string) ?? null,
            finalLoss: (s.result?.final_loss as number) ?? null,
          })
          addToast?.('Turbo training complete!', 'success')
        } else {
          const errMsg = s.error || 'Training failed'
          writeTraining({ phase: 'error', error: errMsg })
          addToast?.(errMsg, 'error')
        }
      } catch {
        clearInterval(pollId); turboPollRef.current = null
      }
    }, 3000)
    turboPollRef.current = pollId
  }, [])

  useEffect(() => () => clearAllPolls(), [clearAllPolls])

  return { startStandardPoll, startTurboPoll, clearAllPolls }
}
