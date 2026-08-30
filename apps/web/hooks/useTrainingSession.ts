'use client'

import { useState, useCallback, useEffect } from 'react'
import { trainingJobsController } from '@/lib/controllers'
import { operationsStore } from '@/lib/operations-store'
import { appShellStore, readTraining, writeTraining, isTrainingActive, type TrainingShellState, type TrainingToastFn } from '@/lib/app-shell'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'
import { extractErrorMessage } from '@/lib/error-utils'
import { useTrainingPolling } from './useTrainingPolling'

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
  avgQuality: number | null
  dataQuality: { avg_quality: number; repetition_rate: number; diversity: number; language_quality: number } | null
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
  pauseTraining: (addToast?: TrainingToastFn) => Promise<void>
  resumeTraining: (addToast?: TrainingToastFn) => Promise<void>
  startFineTune: (params: { model: string; dataset: string; epochs: number; batchSize: number; lr: number; useLoRA: boolean; loraRank?: number; loraAlpha?: number }, addToast: TrainingToastFn, onComplete?: () => void) => void
  startVisualTraining: (params: { dataset: string; visionEncoder: string; llm: string; stage1Epochs: number; stage2Epochs: number; useLoRA: boolean }, addToast: TrainingToastFn, onComplete?: () => void) => void
  startTurboTrain: (datasetId: string, config: { epochs: number; lr: number; embed: number; heads: number; layers: number }, addToast: TrainingToastFn, experimentId?: string) => void
  stopTurboTrain: () => void
}

function useShellTraining(): TrainingShellState {
  const [training, setTraining] = useState<TrainingShellState>(() => readTraining())
  useEffect(() => {
    // Subscribe with shallow equality check to avoid re-renders when
    // the same field values are written (common with batched writes).
    return appShellStore.subscribe((s) => {
      const next = s.training
      setTraining(prev => {
        // Shallow compare — skip update if all fields are equal
        if (prev.phase === next.phase && prev.loss === next.loss
          && prev.progress === next.progress && prev.epoch === next.epoch
          && prev.totalEpochs === next.totalEpochs && prev.globalStep === next.globalStep
          && prev.totalSteps === next.totalSteps && prev.stepsPerSec === next.stepsPerSec
          && prev.eta === next.eta && prev.elapsedSeconds === next.elapsedSeconds
          && prev.jobId === next.jobId && prev.method === next.method
          && prev.error === next.error && prev.message === next.message
          && prev.checkpoint === next.checkpoint && prev.finalLoss === next.finalLoss
          && prev.modelPath === next.modelPath && prev.avgQuality === next.avgQuality
          && prev.lossHistory === next.lossHistory
          && prev.visualOutputDir === next.visualOutputDir && prev.visualSouPath === next.visualSouPath
          && prev.finetunedModelPath === next.finetunedModelPath && prev.finetunedModelLoss === next.finetunedModelLoss
          && prev.dataQuality === next.dataQuality && prev.distillCheckpoint === next.distillCheckpoint
          && prev.distillFinalLoss === next.distillFinalLoss && prev.distillEpochs === next.distillEpochs) {
          return prev
        }
        return next
      })
    })
  }, [])
  return training
}

/**
 * Orchestrator hook for training. Composes useTrainingPolling.
 * Owns: shell subscription, reconciliation, control functions, backward-compat aliases.
 */
export function useTrainingSession(): UseTrainingSessionReturn {
  const training = useShellTraining()
  const { startStandardPoll, startTurboPoll, clearAllPolls } = useTrainingPolling()

  const trainingRunning = isTrainingActive(training)
  const turboRunning = training.method === 'turbo' && trainingRunning

  // Reconcile with server on mount
  useEffect(() => {
    let cancelled = false
    const reconcile = async () => {
      const shell = readTraining()
      if (isTrainingActive(shell)) {
        _log.info('Shell has active training, verifying with server', { phase: shell.phase, method: shell.method })
        // Verify server actually has this training running before trusting localStorage
        try {
          const [turboStatus, jobs] = await Promise.all([
            trainingJobsController.getTurboStatus().catch(() => null),
            shell.jobId ? trainingJobsController.get(shell.jobId).catch(() => null) : Promise.resolve(null),
          ])
          if (cancelled) return

          const turboRunning = turboStatus?.status === 'running'
          const jobRunning = jobs?.status === 'running'

          if (!turboRunning && !jobRunning) {
            // Server has no active training — stale localStorage state, clear it
            _log.info('Shell training is stale (server has no active training), clearing')
            appShellStore.getState().resetTraining()
            useToastStore.getState().addToast('Previous training session expired — server was restarted', 'info')
            return
          }

          // Server confirms training is active — start polling
          if (shell.method === 'turbo' || (shell.method === 'slonet' && !shell.jobId)) {
            startTurboPoll()
          } else if (shell.jobId) {
            startStandardPoll(shell.jobId)
          }
        } catch (e: unknown) {
          _log.warning('Could not verify training state with server', { error: e instanceof Error ? e.message : String(e) })
          // On verification failure, clear stale state to avoid phantom training
          appShellStore.getState().resetTraining()
          useToastStore.getState().addToast('Could not verify training status — showing idle state', 'info')
        }
        return
      }
      try {
        const [turboStatus, jobs] = await Promise.all([
          trainingJobsController.getTurboStatus().catch((e) => { _log.debug('Could not fetch turbo status', { error: e instanceof Error ? e.message : String(e) }); return null }),
          trainingJobsController.list().catch((e) => { _log.debug('Could not fetch training jobs', { error: e instanceof Error ? e.message : String(e) }); return [] }),
        ])
        if (cancelled) return
        if (turboStatus?.status === 'running') {
          _log.info('Server has active turbo training, restoring to shell')
          writeTraining({
            phase: 'TRAINING', method: 'turbo', loss: turboStatus.loss ?? null,
            progress: turboStatus.progress ?? 0, globalStep: turboStatus.global_step ?? 0,
            totalSteps: turboStatus.total_steps ?? 0, stepsPerSec: turboStatus.steps_per_sec ?? null,
            eta: turboStatus.eta_s ?? null, elapsedSeconds: turboStatus.elapsed_s ?? null,
            jobId: turboStatus.job_id ?? null, avgQuality: turboStatus.avg_quality ?? null,
            message: turboStatus.paused ? 'Paused' : '',
          })
          startTurboPoll()
          return
        }
        const runningJob = jobs.find(j => j.status === 'running')
        if (runningJob) {
          _log.info('Server has active standard training, restoring to shell', { jobId: runningJob.id })
          writeTraining({
            phase: 'TRAINING', method: (runningJob.method as TrainingShellState['method']) ?? 'slonet',
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
            modelPath: (turboStatus.result?.model_path as string) ?? null,
            avgQuality: (turboStatus.result?.avg_quality as number) ?? turboStatus.avg_quality ?? null,
            dataQuality: (turboStatus.result?.data_quality as TrainingShellState['dataQuality']) ?? null,
          })
        }
      } catch (e: unknown) { logger.warning('Could not reconcile training service', { exception: String((e instanceof Error ? e.message : e) || e) }) }
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
    appShellStore.getState().resetTraining()
    clearAllPolls()
  }, [clearAllPolls])

  const stopTraining = useCallback(() => {
    trainingJobsController.stopAutoTrain().catch((e) => logger.warning('Could not stop training', e))
    operationsStore.getState().cancelAll('training').catch((e) => logger.warning('Could not cancel all training', { exception: String(e?.message || e) }))
    resetTraining()
  }, [resetTraining])

  const pauseTraining = useCallback(async (addToast?: TrainingToastFn) => {
    try { await trainingJobsController.pauseTraining(); writeTraining({ message: 'Paused' }) } catch (e: unknown) {
      const msg = extractErrorMessage(e, 'Could not pause')
      _log.warning('Could not pause', { error: msg })
      addToast?.(msg, 'error')
      writeTraining({ message: msg })
    }
  }, [])

  const resumeTraining = useCallback(async (addToast?: TrainingToastFn) => {
    try { await trainingJobsController.resumeTraining(); writeTraining({ message: '' }) } catch (e: unknown) {
      const msg = extractErrorMessage(e, 'Could not resume')
      _log.warning('Could not resume', { error: msg })
      addToast?.(msg, 'error')
      writeTraining({ message: msg })
    }
  }, [])

  const startFineTune = useCallback((
    params: { model: string; dataset: string; epochs: number; batchSize: number; lr: number; useLoRA: boolean; loraRank?: number; loraAlpha?: number },
    addToast: TrainingToastFn, onComplete?: () => void,
  ) => {
    appShellStore.getState().resetTraining()
    writeTraining({ phase: 'TRAINING', method: 'hf' })
    trainingJobsController.startLoraFinetune({
      model_path: params.model,
      dataset: params.dataset,
      epochs: params.epochs,
      batch_size: params.batchSize,
      learning_rate: params.lr,
      rank: params.loraRank ?? 8,
      alpha: params.loraAlpha ?? 16.0,
    }).then(resp => {
      const jobId = resp.job_id as string
      addToast('LoRA training queued', 'info')
      writeTraining({ totalEpochs: params.epochs, jobId })
      operationsStore.getState().fetch().catch((e) => _log.debug('Could not fetch operations', { error: e instanceof Error ? e.message : String(e) }))
      startStandardPoll(jobId, { addToast, onComplete })
    }).catch(() => addToast('Something went wrong starting training', 'error'))
  }, [startStandardPoll])

  const startVisualTraining = useCallback((
    params: { dataset: string; visionEncoder: string; llm: string; stage1Epochs: number; stage2Epochs: number; useLoRA: boolean },
    addToast: TrainingToastFn, onComplete?: () => void,
  ) => {
    appShellStore.getState().resetTraining()
    writeTraining({ phase: 'TRAINING', method: 'hf' })
    trainingJobsController.startVisualTrain({
      dataset: params.dataset, vision_encoder: params.visionEncoder, llm: params.llm,
      stage1_epochs: params.stage1Epochs, stage2_epochs: params.stage2Epochs,
      use_lora: params.useLoRA, name: `visual-${params.dataset}-${Date.now()}`,
    }).then(resp => {
      const jobId = resp.job_id as string
      addToast(resp.message || 'Image model training queued', 'info')
      writeTraining({ totalEpochs: params.stage1Epochs + params.stage2Epochs, jobId })
      operationsStore.getState().fetch().catch((e) => _log.debug('Could not fetch operations', { error: e instanceof Error ? e.message : String(e) }))
      startStandardPoll(jobId, {
        addToast, completeMessage: 'Image model training complete',
        onComplete: (job) => {
          writeTraining({
            visualOutputDir: job.output_dir ?? null,
            visualSouPath: job.sou_path ?? null,
          })
          onComplete?.()
        },
      })
    }).catch(() => addToast('Something went wrong starting image model training', 'error'))
  }, [startStandardPoll])

  const startTurboTrain = useCallback((
    datasetId: string, config: { epochs: number; lr: number; embed: number; heads: number; layers: number },
    addToast: TrainingToastFn, experimentId?: string,
  ) => {
    // Request notification permission on first training start
    if (typeof window !== 'undefined' && 'Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission()
    }

    clearAllPolls()
    appShellStore.getState().resetTraining()
    writeTraining({ phase: 'TRAINING', method: 'turbo' })
    trainingJobsController.startTurboTrain({
      dataset_id: datasetId, epochs: config.epochs, learning_rate: config.lr,
      n_embed: config.embed, n_head: config.heads, n_layer: config.layers,
      experiment_id: experimentId,
    }).then(result => {
      if (result.status === 'error') { writeTraining({ error: result.message || 'Could not start turbo training', phase: 'error' }); return }
      addToast('Turbo training started', 'info')
      operationsStore.getState().fetch().catch((e) => _log.debug('Could not fetch operations', { error: e instanceof Error ? e.message : String(e) }))
      startTurboPoll(addToast)
    }).catch((e: unknown) => { writeTraining({ error: extractErrorMessage(e, 'Could not start turbo training'), phase: 'error' }) })
  }, [clearAllPolls, startTurboPoll])

  const stopTurboTrain = useCallback(() => {
    trainingJobsController.stopAutoTrain().catch((e) => logger.warning('Could not stop turbo training', e))
    operationsStore.getState().cancelAll('training').catch((e) => _log.debug('Could not cancel all training ops', { error: e instanceof Error ? e.message : String(e) }))
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
    avgQuality: training.avgQuality ?? null,
    dataQuality: training.dataQuality ?? null,
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
    startFineTune,
    startVisualTraining,
    startTurboTrain,
    stopTurboTrain,
  }
}
