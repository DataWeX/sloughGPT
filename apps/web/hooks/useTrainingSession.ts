'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { trainingJobsController } from '@/lib/controllers'
import { PUBLIC_API_URL } from '@/lib/config'

export interface TrainingSessionState {
  phase: string
  loss: number | null
  progress: number
  epoch: number
  totalEpochs: number
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

  useEffect(() => {
    return () => {
      if (ftPollRef.current) { clearInterval(ftPollRef.current); ftPollRef.current = null }
      if (visualPollRef.current) { clearInterval(visualPollRef.current); visualPollRef.current = null }
    }
  }, [])

  const [phase, setPhase] = useState('idle')
  const [loss, setLoss] = useState<number | null>(null)
  const [progress, setProgress] = useState(0)
  const [epoch, setEpoch] = useState(0)
  const [totalEpochs, setTotalEpochs] = useState(0)
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
    setMessage(''); setLossHistory([]); setEvalResult(null)
  }, [])

  const stopTraining = useCallback(() => {
    esRef.current?.close(); esRef.current = null
    if (ftPollRef.current) { clearInterval(ftPollRef.current); ftPollRef.current = null }
    if (visualPollRef.current) { clearInterval(visualPollRef.current); visualPollRef.current = null }
    trainingJobsController.stopAutoTrain().catch(() => {})
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
        } catch (err) { console.error('[training] SSE parse error:', err) }
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
    trainingJobsController.startHFFineTune({
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
      addToast(resp.message || 'Training queued', 'info')
      setPhase('TRAINING'); setProgress(0); setTotalEpochs(params.epochs)
      const pollId = setInterval(async () => {
        try {
          const resp2 = await fetch(`${PUBLIC_API_URL}/training/jobs`)
          const jobs = await resp2.json()
          const myJob = (jobs || []).find((j: { id: string }) => j.id === jobId)
          if (!myJob) { clearInterval(pollId); ftPollRef.current = null; return }
          if (myJob.status === 'completed') {
            clearInterval(pollId); ftPollRef.current = null
            setPhase('complete'); setProgress(100)
            setFinetunedModelPath(myJob.result?.model_path || '')
            setFinetunedModelLoss(myJob.result?.final_loss || myJob.loss || null)
            addToast('Training complete', 'success')
            onComplete?.()
          } else if (myJob.status === 'failed') {
            clearInterval(pollId); ftPollRef.current = null; setPhase('error')
            addToast(myJob.error || 'Training failed', 'error')
          } else if (myJob.loss != null) {
            setLoss(myJob.loss); setProgress(myJob.progress || 0); setEpoch(myJob.current_epoch || 0)
          }
        } catch { clearInterval(pollId); ftPollRef.current = null }
      }, 3000)
      ftPollRef.current = pollId
      setTimeout(() => { clearInterval(pollId); ftPollRef.current = null }, 300000)
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
          const resp2 = await fetch(`${PUBLIC_API_URL}/training/jobs`)
          const jobs = await resp2.json()
          const myJob = (jobs || []).find((j: { id: string }) => j.id === jobId)
          if (!myJob) {             clearInterval(pollId); visualPollRef.current = null; return }
          if (myJob.status === 'completed') {
            clearInterval(pollId); visualPollRef.current = null; setPhase('complete'); setProgress(100)
            setFinetunedModelPath(myJob.model_path || '')
            setFinetunedModelLoss(myJob.loss || null)
            setVisualOutputDir(myJob.output_dir || null)
            setVisualSouPath(myJob.sou_path || null)
            addToast('Image model training complete', 'success')
            onComplete?.()
          } else if (myJob.status === 'failed') {
            clearInterval(pollId); visualPollRef.current = null; setPhase('error')
            addToast(myJob.error || 'Image model training failed', 'error')
          } else if (myJob.loss != null) {
            setLoss(myJob.loss); setProgress(myJob.progress || 0); setEpoch(myJob.current_epoch || 0)
            setMessage(myJob.stage || '')
          }
        } catch { clearInterval(pollId); visualPollRef.current = null }
      }, 3000)
      visualPollRef.current = pollId
      setTimeout(() => { clearInterval(pollId); visualPollRef.current = null }, 600000)
    }).catch(() => addToast('Something went wrong starting image model training', 'error'))
  }, [])

  const startTurboTrain = useCallback((
    datasetId: string,
    config: { epochs: number; lr: number; embed: number; heads: number; layers: number },
    addToast: (msg: string, type?: 'success' | 'error' | 'info') => void,
  ) => {
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
      } else {
        setTurboResult(result); setTurboPhase('complete')
        addToast('Turbo training complete!', 'success')
      }
    }).catch((e: any) => {
      setTurboError(e?.message || 'Training request failed'); setTurboPhase('error')
    })
  }, [])

  return {
    phase, loss, progress, epoch, totalEpochs, message, startTime, lossHistory, evalResult,
    finetunedModelPath, finetunedModelLoss, distillCheckpoint, distillFinalLoss, distillEpochs,
    turboPhase, turboResult, turboError,
    visualOutputDir, visualSouPath,
    paused,
    setPhase, setLoss, setProgress, setEpoch, setTotalEpochs, setMessage,
    setLossHistory, setEvalResult,
    setFinetunedModelPath, setFinetunedModelLoss,
    setDistillCheckpoint, setDistillFinalLoss, setDistillEpochs,
    setTurboPhase, setTurboResult, setTurboError,
    trainingRunning, turboRunning: turboPhase === 'training',
    resetTraining, stopTraining, pauseTraining, resumeTraining,
    startSSETraining, startFineTune, startVisualTraining, startTurboTrain,
  }
}
