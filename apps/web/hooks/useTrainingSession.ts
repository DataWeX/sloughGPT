'use client'

import { useState, useCallback, useEffect } from 'react'
import { trainingJobsController } from '@/lib/controllers'
import { appShellStore, readTraining, writeTraining, isTrainingActive, type TrainingShellState, type TrainingToastFn } from '@/lib/app-shell'
import { logger } from '@/lib/dev-log'
import { extractErrorMessage } from '@/lib/error-utils'
import { useTrainingPolling } from './useTrainingPolling'
import { useTrainingStream } from './useTrainingStream'

const _log = logger.child('training-session')

export interface UseTrainingSessionReturn extends TrainingShellState {
  trainingRunning: boolean
  turboRunning: boolean
  paused: boolean
  turboPhase: Exclude<TrainingShellState['phase'], 'TRAINING'> | 'training'
  turboResult: { status: string; final_loss: number | null; checkpoint: string | null; model_path: string | null; total_steps: number } | null
  turboError: string | null
  turboProgress: number
  turboGlobalStep: number
  turboTotalSteps: number
  turboStepsPerSec: number | null
  turboEta: number | null
  turboElapsedSeconds: number | null
  turboLoss: number | null
  distillCheckpoint: string | null
  distillFinalLoss: number | null
  distillEpochs: number | null
  finetunedModelPath: string | null
  finetunedModelLoss: number | null
  setPhase: (p: string) => void
  setLoss: (l: number | null) => void
  setProgress: (p: number) => void
  setEpoch: (e: number) => void
  setTotalEpochs: (t: number) => void
  setGlobalStep: (s: number) => void
  setTotalSteps: (s: number) => void
  setEta: (e: number | null) => void
  setStepsPerSec: (s: number | null) => void
  setElapsedSeconds: (e: number | null) => void
  setMessage: (m: string) => void
  setLossHistory: (h: { step: number; loss: number }[]) => void
  setEvalResult: (r: string | null) => void
  resetTraining: () => void
  stopTraining: () => void
  pauseTraining: () => Promise<void>
  resumeTraining: () => Promise<void>
  startSSETraining: (body: Record<string, unknown>, addToast: TrainingToastFn, onCheckpointUpdate?: () => void) => void
  startFineTune: (params: { model: string; dataset: string; epochs: number; batchSize: number; lr: number; useLoRA: boolean }, addToast: TrainingToastFn, onComplete?: () => void) => void
  startVisualTraining: (params: { dataset: string; visionEncoder: string; llm: string; stage1Epochs: number; stage2Epochs: number; useLoRA: boolean }, addToast: TrainingToastFn, onComplete?: () => void) => void
  startTurboTrain: (datasetId: string, config: { epochs: number; lr: number; embed: number; heads: number; layers: number }, addToast: TrainingToastFn) => void
  stopTurboTrain: () => void
}

function useShellTraining(): TrainingShellState {
  const [training, setTraining] = useState<TrainingShellState>(() => readTraining())
  useEffect(() => appShellStore.subscribe((s) => setTraining(s.training)), [])
  return training
}

const IDLE_STATE: Partial<TrainingShellState> = {
  phase: 'idle', method: null, loss: null, progress: 0,
  epoch: 0, totalEpochs: 0, globalStep: 0, totalSteps: 0,
  eta: null, stepsPerSec: null, elapsedSeconds: null,
  message: '', lossHistory: [], evalResult: null,
  checkpoint: null, finalLoss: null, modelPath: null,
  error: null, jobId: null, startTime: null,
  visualOutputDir: null, visualSouPath: null,
}

/**
 * Orchestrator hook for training. Composes useTrainingPolling + useTrainingStream.
 * Owns: shell subscription, reconciliation, control functions, backward-compat aliases.
 */
export function useTrainingSession(): UseTrainingSessionReturn {
  const training = useShellTraining()
  const { startStandardPoll, startTurboPoll, clearAllPolls } = useTrainingPolling()
  const { startSSETraining, closeStream } = useTrainingStream()

  const trainingRunning = isTrainingActive(training)
  const turboRunning = training.method === 'turbo' && trainingRunning

  // Reconcile with server on mount
  useEffect(() => {
    let cancelled = false
    const reconcile = async () => {
      const shell = readTraining()
      if (isTrainingActive(shell)) {
        _log.info('Shell has active training, starting poll', { phase: shell.phase, method: shell.method })
        if (shell.method === 'turbo' && shell.jobId) startTurboPoll()
        else if (shell.jobId) startStandardPoll(shell.jobId)
        return
      }
      try {
        const [turboStatus, jobs] = await Promise.all([
          trainingJobsController.getTurboStatus().catch(() => null),
          trainingJobsController.list().catch(() => []),
        ])
        if (cancelled) return
        if (turboStatus?.status === 'running') {
          _log.info('Server has active turbo training, restoring to shell')
          writeTraining({
            phase: 'TRAINING', method: 'turbo', loss: turboStatus.loss ?? null,
            progress: turboStatus.progress ?? 0, globalStep: turboStatus.global_step ?? 0,
            totalSteps: turboStatus.total_steps ?? 0, stepsPerSec: turboStatus.steps_per_sec ?? null,
            eta: turboStatus.eta_s ?? null, elapsedSeconds: turboStatus.elapsed_s ?? null,
            jobId: turboStatus.job_id ?? null,
          })
          startTurboPoll()
          return
        }
        const runningJob = jobs.find(j => j.status === 'running')
        if (runningJob) {
          _log.info('Server has active standard training, restoring to shell', { jobId: runningJob.id })
          writeTraining({
            phase: 'TRAINING', method: (runningJob.method as TrainingShellState['method']) ?? 'slnet',
            loss: runningJob.loss ?? runningJob.train_loss ?? null, progress: runningJob.progress ?? 0,
            epoch: runningJob.current_epoch ?? 0, totalEpochs: runningJob.epochs ?? 0,
            globalStep: runningJob.global_step ?? 0, totalSteps: runningJob.total_steps ?? 0,
            stepsPerSec: runningJob.steps_per_sec ?? null, eta: runningJob.eta_s ?? null,
            elapsedSeconds: runningJob.elapsed_s ?? null, jobId: runningJob.id,
          })
          startStandardPoll(runningJob.id)
          return
        }
        if (turboStatus?.status === 'complete') {
          writeTraining({
            phase: 'complete', method: 'turbo', progress: 100,
            checkpoint: (turboStatus.result?.checkpoint as string) ?? null,
            finalLoss: (turboStatus.result?.final_loss as number) ?? null,
          })
        }
      } catch { /* Server offline — shell state is still valid */ }
    }
    reconcile()
    return () => { cancelled = true }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Warn before leaving during active training
  useEffect(() => {
    if (!trainingRunning) return
    const handler = (e: BeforeUnloadEvent) => { e.preventDefault() }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [trainingRunning])

  const resetTraining = useCallback(() => {
    writeTraining(IDLE_STATE)
    closeStream()
    clearAllPolls()
  }, [closeStream, clearAllPolls])

  const stopTraining = useCallback(() => {
    trainingJobsController.stopAutoTrain().catch((e) => logger.warning('Failed to stop training', e))
    resetTraining()
  }, [resetTraining])

  const pauseTraining = useCallback(async () => {
    try { await trainingJobsController.pauseTraining(); writeTraining({ message: 'Paused' }) } catch { /* */ }
  }, [])

  const resumeTraining = useCallback(async () => {
    try { await trainingJobsController.resumeTraining(); writeTraining({ message: '' }) } catch { /* */ }
  }, [])

  const startFineTune = useCallback((
    params: { model: string; dataset: string; epochs: number; batchSize: number; lr: number; useLoRA: boolean },
    addToast: TrainingToastFn, onComplete?: () => void,
  ) => {
    writeTraining({ ...IDLE_STATE, phase: 'TRAINING', method: 'hf' })
    trainingJobsController.create({
      model: params.model, dataset: params.dataset, name: `${params.model}-${Date.now()}`,
      epochs: params.epochs, batch_size: params.batchSize, learning_rate: params.lr,
      use_lora: params.useLoRA, lora_rank: 8,
    }).then(resp => {
      const jobId = resp.job_id as string
      addToast('Training queued', 'info')
      writeTraining({ totalEpochs: params.epochs, jobId })
      startStandardPoll(jobId, { addToast, onComplete })
    }).catch(() => addToast('Something went wrong starting training', 'error'))
  }, [startStandardPoll])

  const startVisualTraining = useCallback((
    params: { dataset: string; visionEncoder: string; llm: string; stage1Epochs: number; stage2Epochs: number; useLoRA: boolean },
    addToast: TrainingToastFn, onComplete?: () => void,
  ) => {
    writeTraining({ ...IDLE_STATE, phase: 'TRAINING', method: 'hf' })
    trainingJobsController.startVisualTrain({
      dataset: params.dataset, vision_encoder: params.visionEncoder, llm: params.llm,
      stage1_epochs: params.stage1Epochs, stage2_epochs: params.stage2Epochs,
      use_lora: params.useLoRA, name: `visual-${params.dataset}-${Date.now()}`,
    }).then(resp => {
      const jobId = resp.job_id as string
      addToast(resp.message || 'Image model training queued', 'info')
      writeTraining({ totalEpochs: params.stage1Epochs + params.stage2Epochs, jobId })
      startStandardPoll(jobId, {
        addToast, completeMessage: 'Image model training complete',
        onComplete: (job) => {
          writeTraining({
            visualOutputDir: (job as unknown as Record<string, unknown>)?.output_dir as string ?? null,
            visualSouPath: (job as unknown as Record<string, unknown>)?.sou_path as string ?? null,
          })
          onComplete?.()
        },
      })
    }).catch(() => addToast('Something went wrong starting image model training', 'error'))
  }, [startStandardPoll])

  const startTurboTrain = useCallback((
    datasetId: string, config: { epochs: number; lr: number; embed: number; heads: number; layers: number },
    addToast: TrainingToastFn,
  ) => {
    clearAllPolls()
    writeTraining({ ...IDLE_STATE, phase: 'TRAINING', method: 'turbo' })
    trainingJobsController.startTurboTrain({
      dataset_id: datasetId, epochs: config.epochs, learning_rate: config.lr,
      n_embed: config.embed, n_head: config.heads, n_layer: config.layers,
    }).then(result => {
      if (result.status === 'error') { writeTraining({ error: result.message || 'Training failed', phase: 'error' }); return }
      addToast('Turbo training started', 'info')
      startTurboPoll(addToast)
    }).catch((e: unknown) => { writeTraining({ error: extractErrorMessage(e, 'Training request failed'), phase: 'error' }) })
  }, [clearAllPolls, startTurboPoll])

  const stopTurboTrain = useCallback(() => {
    trainingJobsController.stopAutoTrain().catch((e) => logger.warning('Failed to stop turbo training', e))
    resetTraining()
  }, [resetTraining])

  return {
    ...training,
    trainingRunning,
    turboRunning,
    paused: training.message === 'Paused',
    turboPhase: training.phase === 'TRAINING' ? (training.method === 'turbo' ? 'training' : 'idle') : training.phase,
    turboResult: training.method === 'turbo' ? { status: training.phase, final_loss: training.finalLoss, checkpoint: training.checkpoint, model_path: training.modelPath, total_steps: training.totalSteps } : null,
    turboError: training.error,
    turboProgress: training.progress,
    turboGlobalStep: training.globalStep,
    turboTotalSteps: training.totalSteps,
    turboStepsPerSec: training.stepsPerSec,
    turboEta: training.eta,
    turboElapsedSeconds: training.elapsedSeconds,
    turboLoss: training.loss,
    distillCheckpoint: training.checkpoint,
    distillFinalLoss: training.finalLoss,
    distillEpochs: training.totalEpochs ?? null,
    finetunedModelPath: training.modelPath,
    finetunedModelLoss: training.finalLoss,
    setPhase: (p: string) => writeTraining({ phase: p as TrainingShellState['phase'] }),
    setLoss: (l: number | null) => writeTraining({ loss: l }),
    setProgress: (p: number) => writeTraining({ progress: p }),
    setEpoch: (e: number) => writeTraining({ epoch: e }),
    setTotalEpochs: (t: number) => writeTraining({ totalEpochs: t }),
    setGlobalStep: (s: number) => writeTraining({ globalStep: s }),
    setTotalSteps: (s: number) => writeTraining({ totalSteps: s }),
    setEta: (e: number | null) => writeTraining({ eta: e }),
    setStepsPerSec: (s: number | null) => writeTraining({ stepsPerSec: s }),
    setElapsedSeconds: (e: number | null) => writeTraining({ elapsedSeconds: e }),
    setMessage: (m: string) => writeTraining({ message: m }),
    setLossHistory: (h: { step: number; loss: number }[]) => writeTraining({ lossHistory: h }),
    setEvalResult: (r: string | null) => writeTraining({ evalResult: r }),
    resetTraining,
    stopTraining,
    pauseTraining,
    resumeTraining,
    startSSETraining,
    startFineTune,
    startVisualTraining,
    startTurboTrain,
    stopTurboTrain,
  }
}
