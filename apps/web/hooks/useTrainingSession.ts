'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { trainingJobsController, type TrainingJob } from '@/lib/controllers'
import { PUBLIC_API_URL } from '@/lib/config'
import { logger } from '@/lib/dev-log'
import { extractErrorMessage } from '@/lib/error-utils'

const _log = logger.child('training-session')

export interface TrainingSessionState {
  phase: string
  loss: number | null
  progress: number
  epoch: number
  totalEpochs: number
  globalStep: number
  totalSteps: number
  eta: number | null
  stepsPerSec: number | null
  elapsedSeconds: number | null
  message: string
  startTime: number | null
  lossHistory: { step: number; loss: number }[]
  evalResult: string | null
  finetunedModelPath: string | null
  finetunedModelLoss: number | null
  distillCheckpoint: string | null
  distillFinalLoss: number | null
  distillEpochs: number | null
  turboPhase: 'idle' | 'training' | 'complete' | 'error'
  turboResult: { status: string; final_loss?: number; total_steps?: number; model_path?: string } | null
  turboError: string | null
  visualOutputDir: string | null
  visualSouPath: string | null
}

export interface UseTrainingSessionReturn extends TrainingSessionState {
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
  setFinetunedModelPath: (p: string | null) => void
  setFinetunedModelLoss: (l: number | null) => void
  setDistillCheckpoint: (c: string | null) => void
  setDistillFinalLoss: (l: number | null) => void
  setDistillEpochs: (e: number | null) => void
  setTurboPhase: (p: 'idle' | 'training' | 'complete' | 'error') => void
  setTurboResult: (r: { status: string; final_loss?: number; total_steps?: number; model_path?: string } | null) => void
  setTurboError: (e: string | null) => void
  trainingRunning: boolean
  resetTraining: () => void
  stopTraining: () => void
  pauseTraining: () => Promise<void>
  resumeTraining: () => Promise<void>
  paused: boolean
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
  turboRunning: boolean
}

export function useTrainingSession(): UseTrainingSessionReturn {
  const ftPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const visualPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const turboPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    return () => {
      if (ftPollRef.current) { clearInterval(ftPollRef.current); ftPollRef.current = null }
      if (visualPollRef.current) { clearInterval(visualPollRef.current); visualPollRef.current = null }
      if (turboPollRef.current) { clearInterval(turboPollRef.current); turboPollRef.current = null }
    }
  }, [])

  const [phase, setPhase] = useState('idle')
  const [loss, setLoss] = useState<number | null>(null)
  const [progress, setProgress] = useState(0)
  const [epoch, setEpoch] = useState(0)
  const [totalEpochs, setTotalEpochs] = useState(0)
  const [globalStep, setGlobalStep] = useState(0)
  const [totalSteps, setTotalSteps] = useState(0)
  const [eta, setEta] = useState<number | null>(null)
  const [stepsPerSec, setStepsPerSec] = useState<number | null>(null)
  const [elapsedSeconds, setElapsedSeconds] = useState<number | null>(null)
  const [message, setMessage] = useState('')
  const [startTime, setStartTime] = useState<number | null>(null)
  const [lossHistory, setLossHistory] = useState<{ step: number; loss: number }[]>([])
  const [evalResult, setEvalResult] = useState<string | null>(null)

  const [finetunedModelPath, setFinetunedModelPath] = useState<string | null>(null)
  const [finetunedModelLoss, setFinetunedModelLoss] = useState<number | null>(null)
  const [distillCheckpoint, setDistillCheckpoint] = useState<string | null>(null)
  const [distillFinalLoss, setDistillFinalLoss] = useState<number | null>(null)
  const [distillEpochs, setDistillEpochs] = useState<number | null>(null)

  const [turboPhase, setTurboPhase] = useState<'idle' | 'training' | 'complete' | 'error'>('idle')
  const [turboResult, setTurboResult] = useState<{ status: string; final_loss?: number; total_steps?: number; model_path?: string } | null>(null)
  const [turboError, setTurboError] = useState<string | null>(null)



  const [visualOutputDir, setVisualOutputDir] = useState<string | null>(null)
  const [visualSouPath, setVisualSouPath] = useState<string | null>(null)

  const [paused, setPaused] = useState(false)
  const esRef = useRef<EventSource | null>(null)

  const trainingRunning = phase !== 'idle' && phase !== 'complete' && phase !== 'error'

  const resetTraining = useCallback(() => {
    setPaused(false)
    setFinetunedModelPath(null); setFinetunedModelLoss(null)
    setDistillCheckpoint(null); setDistillFinalLoss(null); setDistillEpochs(null)
    setVisualOutputDir(null); setVisualSouPath(null)
    setPhase('idle'); setProgress(0); setLoss(null); setEpoch(0); setTotalEpochs(0)
    setGlobalStep(0); setTotalSteps(0); setEta(null); setStepsPerSec(null); setElapsedSeconds(null)
    setMessage(''); setLossHistory([]); setEvalResult(null)
  }, [])

  const stopTraining = useCallback(() => {
    esRef.current?.close(); esRef.current = null
    if (ftPollRef.current) { clearInterval(ftPollRef.current); ftPollRef.current = null }
    if (visualPollRef.current) { clearInterval(visualPollRef.current); visualPollRef.current = null }
    trainingJobsController.stopAutoTrain().catch((e) => logger.warning('Failed to stop training', e))
    resetTraining()
  }, [resetTraining])

  const pauseTraining = useCallback(async () => {
    try {
      await trainingJobsController.pauseTraining()
      setPaused(true)
    } catch { /* ignore */ }
  }, [])

  const resumeTraining = useCallback(async () => {
    try {
      await trainingJobsController.resumeTraining()
      setPaused(false)
    } catch { /* ignore */ }
  }, [])

  const startSSETraining = useCallback((
    body: Record<string, unknown>,
    addToast: (msg: string, type?: 'success' | 'error' | 'info') => void,
    onCheckpointUpdate?: () => void,
  ) => {
    esRef.current?.close(); esRef.current = null
    trainingJobsController.startAutoTrain(body).then(() => {
      setPhase('TRAINING'); setProgress(0); setLoss(null); setEpoch(0); setTotalEpochs(0)
      setGlobalStep(0); setTotalSteps(0); setEta(null); setStepsPerSec(null); setElapsedSeconds(null)
      setMessage(''); setLossHistory([]); setEvalResult(null); setStartTime(Date.now())
      const es = new EventSource(`${PUBLIC_API_URL}/auto-train/stream`)
      esRef.current = es
      es.onmessage = (e) => {
        try {
          const env = JSON.parse(e.data)
          if (env.stream !== 'auto-train') return
          setPhase(env.phase || 'TRAINING')
          if (env.data?.loss != null) {
            setLoss(env.data.loss)
            setLossHistory(prev => {
              const last = prev[prev.length - 1]
              const step = (last?.step ?? 0) + 1
              return prev.length > 200 ? prev.slice(-200) : [...prev, { step, loss: env.data.loss }]
            })
          }
          if (env.data?.eval_loss != null) {
            setLossHistory(prev => {
              const step = env.data?.step ?? prev.length
              return prev.length > 200 ? prev.slice(-200) : [...prev, { step, loss: env.data.eval_loss, isEval: true }]
            })
          }
          if (env.data?.progress != null) setProgress(env.data.progress)
          if (env.data?.global_step != null) setGlobalStep(env.data.global_step)
          if (env.data?.total_steps != null) setTotalSteps(env.data.total_steps)
          if (env.data?.eta_s != null) setEta(env.data.eta_s)
          if (env.data?.steps_per_sec != null) setStepsPerSec(env.data.steps_per_sec)
          if (env.data?.elapsed_s != null) setElapsedSeconds(env.data.elapsed_s)
          if (env.meta?.epoch != null) setEpoch(env.meta.epoch)
          if (env.meta?.total_epochs != null) setTotalEpochs(env.meta.total_epochs)
          if (env.message) setMessage(env.message)
          if (env.data?.eval_report) setEvalResult(env.data.eval_report)
          if (env.data?.done && env.data?.done_reason?.startsWith('early_stopping:')) {
            const n = env.data.done_reason.split(':')[1]
            addToast(`Early stopping: no improvement for ${n} evals`, 'info')
          }
          if (env.status === 'complete') {
            es.close(); esRef.current = null
            if (env.data?.checkpoint) setDistillCheckpoint(env.data.checkpoint)
            if (env.data?.final_loss != null) setDistillFinalLoss(env.data.final_loss)
            if (env.data?.epochs != null) setDistillEpochs(env.data.epochs)
            setPhase('complete')
            addToast('Training complete', 'success')
            onCheckpointUpdate?.()
          }
          if (env.status === 'error') { es.close(); esRef.current = null; setPhase('error'); addToast('Training failed', 'error') }
        } catch (err) { _log.error('SSE parse error', { exception: String(err) }) }
      }
      let esRetries = 0
      es.onerror = () => {
        if (es.readyState === EventSource.CLOSED || esRetries >= 3) {
          es.close(); esRef.current = null; setPhase('error')
          addToast('Connection lost during training', 'error')
        } else { esRetries++ }
      }
    }).catch(() => addToast('Failed to start training', 'error'))
  }, [])

  const startFineTune = useCallback((
    params: { model: string; dataset: string; epochs: number; batchSize: number; lr: number; useLoRA: boolean },
    addToast: (msg: string, type?: 'success' | 'error' | 'info') => void,
    onComplete?: () => void,
  ) => {
    setFinetunedModelPath(null); setFinetunedModelLoss(null)
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
      setPhase('TRAINING'); setProgress(0); setTotalEpochs(params.epochs)
      const pollId = setInterval(async () => {
        try {
          const jobs = await trainingJobsController.list()
          const myJob = (jobs || []).find((j: { id: string }) => j.id === jobId)
          if (!myJob) { clearInterval(pollId); ftPollRef.current = null; return }
          if (myJob.status === 'completed') {
            clearInterval(pollId); ftPollRef.current = null
            setPhase('complete'); setProgress(100)
            const result = myJob.result as Record<string, unknown> | undefined
            setFinetunedModelPath((result?.model_path as string) || '')
            setFinetunedModelLoss((result?.final_loss as number) ?? myJob.loss ?? null)
            addToast('Training complete', 'success')
            onComplete?.()
          } else if (myJob.status === 'failed') {
            clearInterval(pollId); ftPollRef.current = null; setPhase('error')
            addToast(myJob.error || 'Training failed', 'error')
          } else if (myJob.loss != null) {
            setLoss(myJob.loss); setProgress(myJob.progress || 0); setEpoch(myJob.current_epoch || 0)
            if (myJob.global_step != null) setGlobalStep(myJob.global_step)
            if (myJob.total_steps != null) setTotalSteps(myJob.total_steps)
            if (myJob.eta_s != null) setEta(myJob.eta_s)
            if (myJob.steps_per_sec != null) setStepsPerSec(myJob.steps_per_sec)
            if (myJob.elapsed_s != null) setElapsedSeconds(myJob.elapsed_s)
          }
        } catch { clearInterval(pollId); ftPollRef.current = null }
      }, 3000)
      ftPollRef.current = pollId
    }).catch(() => addToast('Something went wrong starting training', 'error'))
  }, [])

  const startVisualTraining = useCallback((
    params: { dataset: string; visionEncoder: string; llm: string; stage1Epochs: number; stage2Epochs: number; useLoRA: boolean },
    addToast: (msg: string, type?: 'success' | 'error' | 'info') => void,
    onComplete?: () => void,
  ) => {
    setFinetunedModelPath(null); setFinetunedModelLoss(null)
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
      setPhase('TRAINING'); setProgress(0); setTotalEpochs(params.stage1Epochs + params.stage2Epochs)
      const pollId = setInterval(async () => {
        try {
          const jobs = await trainingJobsController.list()
          const myJob = (jobs || []).find((j: { id: string }) => j.id === jobId) as Record<string, unknown> | undefined
          if (!myJob) {             clearInterval(pollId); visualPollRef.current = null; return }
          if (myJob.status === 'completed') {
            clearInterval(pollId); visualPollRef.current = null; setPhase('complete'); setProgress(100)
            setFinetunedModelPath((myJob.model_path as string) || '')
            setFinetunedModelLoss((myJob.loss as number) || null)
            setVisualOutputDir((myJob.output_dir as string) || null)
            setVisualSouPath((myJob.sou_path as string) || null)
            addToast('Image model training complete', 'success')
            onComplete?.()
          } else if (myJob.status === 'failed') {
            clearInterval(pollId); visualPollRef.current = null; setPhase('error')
            addToast((myJob.error as string) || 'Image model training failed', 'error')
          } else if (myJob.loss != null) {
            setLoss(myJob.loss as number); setProgress((myJob.progress as number) || 0); setEpoch((myJob.current_epoch as number) || 0)
            setMessage((myJob.stage as string) || '')
          }
        } catch { clearInterval(pollId); visualPollRef.current = null }
      }, 3000)
      visualPollRef.current = pollId
    }).catch(() => addToast('Something went wrong starting image model training', 'error'))
  }, [])

  const startTurboTrain = useCallback((
    datasetId: string,
    config: { epochs: number; lr: number; embed: number; heads: number; layers: number },
    addToast: (msg: string, type?: 'success' | 'error' | 'info') => void,
  ) => {
    if (turboPollRef.current) { clearInterval(turboPollRef.current); turboPollRef.current = null }
    setTurboPhase('training'); setTurboResult(null); setTurboError(null)
    trainingJobsController.startTurboTrain({
      dataset_id: datasetId,
      epochs: config.epochs,
      learning_rate: config.lr,
      n_embed: config.embed,
      n_head: config.heads,
      n_layer: config.layers,
    }).then(result => {
      if (result.status === 'error') {
        setTurboError(result.message || 'Training failed'); setTurboPhase('error')
        return
      }
      addToast('Turbo training queued', 'info')
      const pollId = setInterval(async () => {
        try {
          const status = await trainingJobsController.getTurboStatus()
          if (status.status === 'running' || status.status === 'idle') {
            if (status.progress != null) setProgress(status.progress)
            if (status.loss != null) setLoss(status.loss)
            if (status.global_step != null) setGlobalStep(status.global_step)
            if (status.total_steps != null) setTotalSteps(status.total_steps)
            if (status.steps_per_sec != null) setStepsPerSec(status.steps_per_sec)
            if (status.eta_s != null) setEta(status.eta_s)
            if (status.elapsed_s != null) setElapsedSeconds(status.elapsed_s)
            return
          }
          clearInterval(pollId); turboPollRef.current = null
          if (status.status === 'complete') {
            setProgress(100)
            setTurboResult((status.result as { status: string; final_loss?: number; total_steps?: number; model_path?: string } | null) ?? { status: 'complete' })
            setTurboPhase('complete')
            addToast('Turbo training complete!', 'success')
          } else {
            setTurboError(status.error || 'Turbo training failed'); setTurboPhase('error')
            addToast(status.error || 'Turbo training failed', 'error')
          }
        } catch {
          clearInterval(pollId); turboPollRef.current = null
          setTurboError('Failed to check turbo training status'); setTurboPhase('error')
        }
      }, 3000)
      turboPollRef.current = pollId
    }).catch((e: unknown) => {
      setTurboError(extractErrorMessage(e, 'Training request failed')); setTurboPhase('error')
    })
  }, [])

  return {
    phase, loss, progress, epoch, totalEpochs, globalStep, totalSteps, eta, stepsPerSec, elapsedSeconds,
    message, startTime, lossHistory, evalResult,
    finetunedModelPath, finetunedModelLoss, distillCheckpoint, distillFinalLoss, distillEpochs,
    turboPhase, turboResult, turboError,
    visualOutputDir, visualSouPath,
    paused,
    setPhase, setLoss, setProgress, setEpoch, setTotalEpochs,
    setGlobalStep, setTotalSteps, setEta, setStepsPerSec, setElapsedSeconds,
    setMessage,
    setLossHistory, setEvalResult,
    setFinetunedModelPath, setFinetunedModelLoss,
    setDistillCheckpoint, setDistillFinalLoss, setDistillEpochs,
    setTurboPhase, setTurboResult, setTurboError,
    trainingRunning, turboRunning: turboPhase === 'training',
    resetTraining, stopTraining, pauseTraining, resumeTraining,
    startSSETraining, startFineTune, startVisualTraining, startTurboTrain,
  }
}
