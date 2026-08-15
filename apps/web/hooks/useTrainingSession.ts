'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { trainingJobsController, type TrainingJob } from '@/lib/controllers'
import { appShellStore, isTrainingActive, type TrainingShellState } from '@/lib/app-shell'
import { PUBLIC_API_URL } from '@/lib/config'
import { logger } from '@/lib/dev-log'
import { extractErrorMessage } from '@/lib/error-utils'

const _log = logger.child('training-session')

export interface UseTrainingSessionReturn extends TrainingShellState {
  trainingRunning: boolean
  turboRunning: boolean
  paused: boolean
  // Backward-compatible aliases for turbo/distill/finetune
  turboPhase: Exclude<TrainingShellState['phase'], 'TRAINING'> | 'training'
  turboResult: {
    status: string
    final_loss: number | null
    checkpoint: string | null
    model_path: string | null
    total_steps: number
  } | null
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
  // Backward-compatible setters
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
  startSSETraining: (body: Record<string, unknown>, addToast: (msg: string, type?: 'success' | 'error' | 'info') => void, onCheckpointUpdate?: () => void) => void
  startFineTune: (params: {
    model: string; dataset: string; epochs: number; batchSize: number; lr: number; useLoRA: boolean
  }, addToast: (msg: string, type?: 'success' | 'error' | 'info') => void, onComplete?: () => void) => void
  startVisualTraining: (params: {
    dataset: string; visionEncoder: string; llm: string; stage1Epochs: number; stage2Epochs: number; useLoRA: boolean
  }, addToast: (msg: string, type?: 'success' | 'error' | 'info') => void, onComplete?: () => void) => void
  startTurboTrain: (datasetId: string, config: {
    epochs: number; lr: number; embed: number; heads: number; layers: number
  }, addToast: (msg: string, type?: 'success' | 'error' | 'info') => void) => void
  stopTurboTrain: () => void
}

/**
 * Subscribe to appShellStore.training and return it as React state.
 * The shell is the source of truth — this hook bridges it to React.
 */
function useShellTraining(): TrainingShellState {
  const [training, setTraining] = useState<TrainingShellState>(() => appShellStore.getState().training)

  useEffect(() => {
    return appShellStore.subscribe((state) => {
      setTraining(state.training)
    })
  }, [])

  return training
}

export function useTrainingSession(): UseTrainingSessionReturn {
  const ftPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const visualPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const turboPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const esRef = useRef<EventSource | null>(null)

  // Shell is the source of truth — subscribe to it
  const training = useShellTraining()
  const set = useCallback((partial: Partial<TrainingShellState>) => {
    appShellStore.getState().setTraining(partial)
  }, [])

  const trainingRunning = isTrainingActive(training)
  const turboRunning = training.method === 'turbo' && trainingRunning

  // Reconcile with server on mount — detect training that started before this page load
  useEffect(() => {
    let cancelled = false
    const reconcile = async () => {
      // If shell already has active training, just start polling
      if (isTrainingActive(appShellStore.getState().training)) {
        const shell = appShellStore.getState().training
        _log.info('Shell has active training, starting poll', { phase: shell.phase, method: shell.method })
        if (shell.method === 'turbo' && shell.jobId) {
          startTurboPoll()
        } else if (shell.jobId) {
          startStandardPoll(shell.jobId)
        }
        return
      }

      // Check server for any running training (turbo OR standard)
      try {
        const [turboStatus, jobs] = await Promise.all([
          trainingJobsController.getTurboStatus().catch(() => null),
          trainingJobsController.list().catch(() => []),
        ])
        if (cancelled) return

        // Check turbo first
        if (turboStatus?.status === 'running') {
          _log.info('Server has active turbo training, restoring to shell')
          set({
            phase: 'TRAINING',
            method: 'turbo',
            loss: turboStatus.loss ?? null,
            progress: turboStatus.progress ?? 0,
            globalStep: turboStatus.global_step ?? 0,
            totalSteps: turboStatus.total_steps ?? 0,
            stepsPerSec: turboStatus.steps_per_sec ?? null,
            eta: turboStatus.eta_s ?? null,
            elapsedSeconds: turboStatus.elapsed_s ?? null,
            jobId: turboStatus.job_id ?? null,
          })
          startTurboPoll()
          return
        }

        // Check standard training jobs
        const runningJob = jobs.find(j => j.status === 'running')
        if (runningJob) {
          _log.info('Server has active standard training, restoring to shell', { jobId: runningJob.id })
          set({
            phase: 'TRAINING',
            method: (runningJob.method as TrainingShellState['method']) ?? 'slnet',
            loss: runningJob.loss ?? runningJob.train_loss ?? null,
            progress: runningJob.progress ?? 0,
            epoch: runningJob.current_epoch ?? 0,
            totalEpochs: runningJob.epochs ?? 0,
            globalStep: runningJob.global_step ?? 0,
            totalSteps: runningJob.total_steps ?? 0,
            stepsPerSec: runningJob.steps_per_sec ?? null,
            eta: runningJob.eta_s ?? null,
            elapsedSeconds: runningJob.elapsed_s ?? null,
            jobId: runningJob.id,
          })
          startStandardPoll(runningJob.id)
          return
        }

        // Check for completed turbo
        if (turboStatus?.status === 'complete') {
          set({
            phase: 'complete',
            method: 'turbo',
            progress: 100,
            checkpoint: (turboStatus.result?.checkpoint as string) ?? null,
            finalLoss: (turboStatus.result?.final_loss as number) ?? null,
          })
        }
      } catch {
        // Server offline — shell state is still valid
      }
    }
    reconcile()
    return () => { cancelled = true }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Warn before leaving during active training
  useEffect(() => {
    if (!trainingRunning) return
    const onBeforeUnload = (e: BeforeUnloadEvent) => { e.preventDefault() }
    window.addEventListener('beforeunload', onBeforeUnload)
    return () => window.removeEventListener('beforeunload', onBeforeUnload)
  }, [trainingRunning])

  // Keyboard shortcuts for training page
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault()
        // Handled by parent
      }
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'T') {
        e.preventDefault()
        // Handled by parent
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  useEffect(() => {
    return () => {
      if (ftPollRef.current) { clearInterval(ftPollRef.current); ftPollRef.current = null }
      if (visualPollRef.current) { clearInterval(visualPollRef.current); visualPollRef.current = null }
      if (turboPollRef.current) { clearInterval(turboPollRef.current); turboPollRef.current = null }
    }
  }, [])

  const resetTraining = useCallback(() => {
    set({ ...appShellStore.getState().training, phase: 'idle', method: null, loss: null, progress: 0, epoch: 0, totalEpochs: 0, globalStep: 0, totalSteps: 0, eta: null, stepsPerSec: null, elapsedSeconds: null, message: '', lossHistory: [], evalResult: null, checkpoint: null, finalLoss: null, modelPath: null, error: null, jobId: null })
    esRef.current?.close(); esRef.current = null
    if (ftPollRef.current) { clearInterval(ftPollRef.current); ftPollRef.current = null }
    if (visualPollRef.current) { clearInterval(visualPollRef.current); visualPollRef.current = null }
    if (turboPollRef.current) { clearInterval(turboPollRef.current); turboPollRef.current = null }
  }, [set])

  const stopTraining = useCallback(() => {
    esRef.current?.close(); esRef.current = null
    if (ftPollRef.current) { clearInterval(ftPollRef.current); ftPollRef.current = null }
    if (visualPollRef.current) { clearInterval(visualPollRef.current); visualPollRef.current = null }
    if (turboPollRef.current) { clearInterval(turboPollRef.current); turboPollRef.current = null }
    trainingJobsController.stopAutoTrain().catch((e) => logger.warning('Failed to stop training', e))
    resetTraining()
  }, [resetTraining])

  const pauseTraining = useCallback(async () => {
    try {
      await trainingJobsController.pauseTraining()
      set({ message: 'Paused' })
    } catch { /* ignore */ }
  }, [set])

  const resumeTraining = useCallback(async () => {
    try {
      await trainingJobsController.resumeTraining()
      set({ message: '' })
    } catch { /* ignore */ }
  }, [set])

  function startTurboPoll(addToast?: (msg: string, type?: 'success' | 'error' | 'info') => void) {
    if (turboPollRef.current) { clearInterval(turboPollRef.current); turboPollRef.current = null }
    const pollId = setInterval(async () => {
      try {
        const s = await trainingJobsController.getTurboStatus()
        const current = appShellStore.getState().training
        if (s.status === 'running' || s.status === 'idle') {
          set({
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
          set({
            phase: 'complete',
            progress: 100,
            checkpoint: (s.result?.checkpoint as string) ?? null,
            finalLoss: (s.result?.final_loss as number) ?? null,
          })
          addToast?.('Turbo training complete!', 'success')
        } else {
          const errMsg = s.error || 'Training failed'
          set({ phase: 'error', error: errMsg })
          addToast?.(errMsg, 'error')
        }
      } catch {
        clearInterval(pollId); turboPollRef.current = null
      }
    }, 3000)
    turboPollRef.current = pollId
  }

  function startStandardPoll(jobId: string) {
    if (ftPollRef.current) { clearInterval(ftPollRef.current); ftPollRef.current = null }
    const pollId = setInterval(async () => {
      try {
        const job = await trainingJobsController.get(jobId)
        if (!job) { clearInterval(pollId); ftPollRef.current = null; return }
        const current = appShellStore.getState().training
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
          set(patch)
          return
        }
        clearInterval(pollId); ftPollRef.current = null
        if (job.status === 'completed') {
          set({
            phase: 'complete',
            progress: 100,
            checkpoint: job.checkpoint ?? null,
            finalLoss: job.loss ?? job.train_loss ?? null,
            modelPath: job.result?.model_path as string ?? null,
          })
        } else {
          const errMsg = job.error || 'Training failed'
          set({ phase: 'error', error: errMsg })
        }
      } catch {
        clearInterval(pollId); ftPollRef.current = null
      }
    }, 3000)
    ftPollRef.current = pollId
  }

  const startSSETraining = useCallback((
    body: Record<string, unknown>,
    addToast: (msg: string, type?: 'success' | 'error' | 'info') => void,
    onCheckpointUpdate?: () => void,
  ) => {
    esRef.current?.close(); esRef.current = null
    trainingJobsController.startAutoTrain(body).then(() => {
      set({ phase: 'TRAINING', method: 'slnet', progress: 0, loss: null, epoch: 0, totalEpochs: 0, globalStep: 0, totalSteps: 0, eta: null, stepsPerSec: null, elapsedSeconds: null, message: '', lossHistory: [], evalResult: null, startTime: Date.now(), error: null })
      const es = new EventSource(`${PUBLIC_API_URL}/auto-train/stream`)
      esRef.current = es
      es.onmessage = (e) => {
        try {
          const env = JSON.parse(e.data)
          if (env.stream !== 'auto-train') return
          set({ phase: env.phase || 'TRAINING' })
          if (env.data?.loss != null) {
            const current = appShellStore.getState().training
            const hist = [...current.lossHistory, { step: (current.lossHistory[current.lossHistory.length - 1]?.step ?? 0) + 1, loss: env.data.loss }]
            set({ loss: env.data.loss, lossHistory: hist.length > 200 ? hist.slice(-200) : hist })
          }
          if (env.data?.eval_loss != null) {
            const current = appShellStore.getState().training
            const hist = [...current.lossHistory, { step: env.data?.step ?? current.lossHistory.length, loss: env.data.eval_loss, isEval: true }]
            set({ lossHistory: hist.length > 200 ? hist.slice(-200) : hist })
          }
          if (env.data?.progress != null) set({ progress: env.data.progress })
          if (env.data?.global_step != null) set({ globalStep: env.data.global_step })
          if (env.data?.total_steps != null) set({ totalSteps: env.data.total_steps })
          if (env.data?.eta_s != null) set({ eta: env.data.eta_s })
          if (env.data?.steps_per_sec != null) set({ stepsPerSec: env.data.steps_per_sec })
          if (env.data?.elapsed_s != null) set({ elapsedSeconds: env.data.elapsed_s })
          if (env.meta?.epoch != null) set({ epoch: env.meta.epoch })
          if (env.meta?.total_epochs != null) set({ totalEpochs: env.meta.total_epochs })
          if (env.message) set({ message: env.message })
          if (env.data?.eval_report) set({ evalResult: env.data.eval_report })
          if (env.status === 'complete') {
            es.close(); esRef.current = null
            set({ phase: 'complete', checkpoint: env.data?.checkpoint ?? null, finalLoss: env.data?.final_loss ?? null })
            addToast('Training complete', 'success')
            onCheckpointUpdate?.()
          }
          if (env.status === 'error') { es.close(); esRef.current = null; set({ phase: 'error', error: 'Training failed' }); addToast('Training failed', 'error') }
        } catch (err) { _log.error('SSE parse error', { exception: String(err) }) }
      }
      let esRetries = 0
      es.onerror = () => {
        if (es.readyState === EventSource.CLOSED || esRetries >= 3) {
          es.close(); esRef.current = null; set({ phase: 'error', error: 'Connection lost' })
          addToast('Connection lost during training', 'error')
        } else { esRetries++ }
      }
    }).catch(() => addToast('Failed to start training', 'error'))
  }, [set])

  const startFineTune = useCallback((
    params: { model: string; dataset: string; epochs: number; batchSize: number; lr: number; useLoRA: boolean },
    addToast: (msg: string, type?: 'success' | 'error' | 'info') => void,
    onComplete?: () => void,
  ) => {
    set({ modelPath: null, finalLoss: null })
    trainingJobsController.create({
      model: params.model,
      dataset: params.dataset,
      name: `${params.model}-${Date.now()}`,
      epochs: params.epochs,
      batch_size: params.batchSize,
      learning_rate: params.lr,
      use_lora: params.useLoRA,
      lora_rank: 8,
    }).then(resp => {
      const jobId = resp.job_id
      addToast('Training queued', 'info')
      set({ phase: 'TRAINING', method: 'hf', progress: 0, totalEpochs: params.epochs, jobId })
      const pollId = setInterval(async () => {
        try {
          const jobs = await trainingJobsController.list()
          const myJob = (jobs || []).find((j: { id: string }) => j.id === jobId)
          if (!myJob) { clearInterval(pollId); ftPollRef.current = null; return }
          if (myJob.status === 'completed') {
            clearInterval(pollId); ftPollRef.current = null
            const result = myJob.result as Record<string, unknown> | undefined
            set({ phase: 'complete', progress: 100, modelPath: (result?.model_path as string) || '', finalLoss: (result?.final_loss as number) ?? myJob.loss ?? null })
            addToast('Training complete', 'success')
            onComplete?.()
          } else if (myJob.status === 'failed') {
            clearInterval(pollId); ftPollRef.current = null; set({ phase: 'error', error: myJob.error || 'Training failed' })
            addToast(myJob.error || 'Training failed', 'error')
          } else if (myJob.loss != null) {
            const current = appShellStore.getState().training
            set({ loss: myJob.loss, progress: myJob.progress || 0, epoch: myJob.current_epoch || 0, globalStep: myJob.global_step ?? current.globalStep, totalSteps: myJob.total_steps ?? current.totalSteps, eta: myJob.eta_s ?? current.eta, stepsPerSec: myJob.steps_per_sec ?? current.stepsPerSec, elapsedSeconds: myJob.elapsed_s ?? current.elapsedSeconds })
          }
        } catch { clearInterval(pollId); ftPollRef.current = null }
      }, 3000)
      ftPollRef.current = pollId
    }).catch(() => addToast('Something went wrong starting training', 'error'))
  }, [set])

  const startVisualTraining = useCallback((
    params: { dataset: string; visionEncoder: string; llm: string; stage1Epochs: number; stage2Epochs: number; useLoRA: boolean },
    addToast: (msg: string, type?: 'success' | 'error' | 'info') => void,
    onComplete?: () => void,
  ) => {
    set({ modelPath: null, finalLoss: null })
    trainingJobsController.startVisualTrain({
      dataset: params.dataset,
      vision_encoder: params.visionEncoder,
      llm: params.llm,
      stage1_epochs: params.stage1Epochs,
      stage2_epochs: params.stage2Epochs,
      use_lora: params.useLoRA,
      name: `visual-${params.dataset}-${Date.now()}`,
    }).then(resp => {
      const jobId = resp.job_id
      addToast(resp.message || 'Image model training queued', 'info')
      set({ phase: 'TRAINING', method: 'hf', progress: 0, totalEpochs: params.stage1Epochs + params.stage2Epochs, jobId })
      const pollId = setInterval(async () => {
        try {
          const jobs = await trainingJobsController.list()
          const myJob = (jobs || []).find((j: { id: string }) => j.id === jobId) as Record<string, unknown> | undefined
          if (!myJob) { clearInterval(pollId); visualPollRef.current = null; return }
          if (myJob.status === 'completed') {
            clearInterval(pollId); visualPollRef.current = null
            set({ phase: 'complete', progress: 100, modelPath: (myJob.model_path as string) || '', finalLoss: (myJob.loss as number) || null, visualOutputDir: (myJob.output_dir as string) || null, visualSouPath: (myJob.sou_path as string) || null })
            addToast('Image model training complete', 'success')
            onComplete?.()
          } else if (myJob.status === 'failed') {
            clearInterval(pollId); visualPollRef.current = null; set({ phase: 'error', error: (myJob.error as string) || 'Image model training failed' })
            addToast((myJob.error as string) || 'Image model training failed', 'error')
          } else if (myJob.loss != null) {
            set({ loss: myJob.loss as number, progress: (myJob.progress as number) || 0, epoch: (myJob.current_epoch as number) || 0, message: (myJob.stage as string) || '' })
          }
        } catch { clearInterval(pollId); visualPollRef.current = null }
      }, 3000)
      visualPollRef.current = pollId
    }).catch(() => addToast('Something went wrong starting image model training', 'error'))
  }, [set])

  const startTurboTrain = useCallback((
    datasetId: string,
    config: { epochs: number; lr: number; embed: number; heads: number; layers: number },
    addToast: (msg: string, type?: 'success' | 'error' | 'info') => void,
  ) => {
    if (turboPollRef.current) { clearInterval(turboPollRef.current); turboPollRef.current = null }
    set({ phase: 'TRAINING', method: 'turbo', checkpoint: null, finalLoss: null, error: null, progress: 0, globalStep: 0, totalSteps: 0, eta: null, stepsPerSec: null, elapsedSeconds: null, loss: null })
    trainingJobsController.startTurboTrain({
      dataset_id: datasetId,
      epochs: config.epochs,
      learning_rate: config.lr,
      n_embed: config.embed,
      n_head: config.heads,
      n_layer: config.layers,
    }).then(result => {
      if (result.status === 'error') {
        set({ error: result.message || 'Training failed', phase: 'error' })
        return
      }
      addToast('Turbo training started', 'info')
      startTurboPoll(addToast)
    }).catch((e: unknown) => {
      set({ error: extractErrorMessage(e, 'Training request failed'), phase: 'error' })
    })
  }, [set])

  const stopTurboTrain = useCallback(() => {
    if (turboPollRef.current) { clearInterval(turboPollRef.current); turboPollRef.current = null }
    trainingJobsController.stopAutoTrain().catch((e) => logger.warning('Failed to stop turbo training', e))
    set({ phase: 'idle', method: null, checkpoint: null, finalLoss: null, error: null, progress: 0, globalStep: 0, totalSteps: 0, eta: null, stepsPerSec: null, elapsedSeconds: null, loss: null })
  }, [set])

  return {
    ...training,
    trainingRunning,
    turboRunning,
    paused: training.message === 'Paused',
    // Backward-compatible aliases
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
    distillEpochs: training.totalEpochs || null,
    finetunedModelPath: training.modelPath,
    finetunedModelLoss: training.finalLoss,
    visualOutputDir: training.visualOutputDir,
    visualSouPath: training.visualSouPath,
    // Backward-compatible setters (delegate to shell)
    setPhase: (p: string) => set({ phase: p as TrainingShellState['phase'] }),
    setLoss: (l: number | null) => set({ loss: l }),
    setProgress: (p: number) => set({ progress: p }),
    setEpoch: (e: number) => set({ epoch: e }),
    setTotalEpochs: (t: number) => set({ totalEpochs: t }),
    setGlobalStep: (s: number) => set({ globalStep: s }),
    setTotalSteps: (s: number) => set({ totalSteps: s }),
    setEta: (e: number | null) => set({ eta: e }),
    setStepsPerSec: (s: number | null) => set({ stepsPerSec: s }),
    setElapsedSeconds: (e: number | null) => set({ elapsedSeconds: e }),
    setMessage: (m: string) => set({ message: m }),
    setLossHistory: (h: { step: number; loss: number }[]) => set({ lossHistory: h }),
    setEvalResult: (r: string | null) => set({ evalResult: r }),
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
