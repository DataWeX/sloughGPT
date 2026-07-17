'use client'

import { useCallback, useEffect, useState } from 'react'
import { modelController } from '@/lib/controllers'
import type { TrainingJob } from '@/lib/training-controller'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { UseTrainingSessionReturn } from '@/hooks/useTrainingSession'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'
import { chatDB } from '@/lib/db'

export type Method = 'distill' | 'finetune' | 'vlm'
export type InputMode = 'dataset' | 'text'

export interface TrainingFormState {
  method: Method
  inputMode: InputMode
  textInput: string
  showAdvanced: boolean
  algo: string
  trainingEpochs: number
  trainingLR: number
  trainingBatchSize: number
  availableModels: string[]
  selectedModel: string
  useLoRA: boolean
  visualVisionEncoder: string
  visualLLM: string
  visualStage1Epochs: number
  visualStage2Epochs: number
  loadingFinetunedModel: boolean
  allJobs: TrainingJob[]
  setMethod: (m: Method) => void
  setInputMode: (m: InputMode) => void
  setTextInput: (s: string) => void
  setShowAdvanced: (v: boolean) => void
  setAlgo: (a: string) => void
  setTrainingEpochs: (n: number) => void
  setTrainingLR: (n: number) => void
  setTrainingBatchSize: (n: number) => void
  setSelectedModel: (s: string) => void
  setUseLoRA: (v: boolean) => void
  setVlmVisionEncoder: (s: string) => void
  setVlmLLM: (s: string) => void
  setVlmStage1Epochs: (n: number) => void
  setVlmStage2Epochs: (n: number) => void
  setLoadingFinetunedModel: (v: boolean) => void
  canStart: boolean
  startTraining: (checkpointName?: string) => void
}

const TRAINING_CONFIG_KEY = 'sloughgpt-training-config'

interface SavedConfig {
  method?: Method
  inputMode?: InputMode
  algo?: string
  trainingEpochs?: number
  trainingLR?: number
  trainingBatchSize?: number
  selectedModel?: string
  useLoRA?: boolean
}

export function useTrainingForm(
  datasets: UseTrainingDatasetsReturn,
  session: UseTrainingSessionReturn,
  checkpoints: UseTrainingCheckpointsReturn,
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void,
): TrainingFormState {
  const [method, setMethod] = useState<Method>('distill')
  const [inputMode, setInputMode] = useState<InputMode>('dataset')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [algo, setAlgo] = useState('bpe')
  const [trainingEpochs, setTrainingEpochs] = useState(5)
  const [trainingLR, setTrainingLR] = useState(1e-3)
  const [trainingBatchSize, setTrainingBatchSize] = useState(64)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [useLoRA, setUseLoRA] = useState(true)
  const [textInput, setTextInput] = useState('')
  const [configLoaded, setConfigLoaded] = useState(false)

  useEffect(() => {
    chatDB.getKV<SavedConfig>(TRAINING_CONFIG_KEY).then(saved => {
      if (saved) {
        if (saved.method) setMethod(saved.method)
        if (saved.inputMode) setInputMode(saved.inputMode)
        if (saved.algo) setAlgo(saved.algo)
        if (saved.trainingEpochs) setTrainingEpochs(saved.trainingEpochs)
        if (saved.trainingLR) setTrainingLR(saved.trainingLR)
        if (saved.trainingBatchSize) setTrainingBatchSize(saved.trainingBatchSize)
        if (saved.selectedModel) setSelectedModel(saved.selectedModel)
        if (saved.useLoRA !== undefined) setUseLoRA(saved.useLoRA)
      }
      setConfigLoaded(true)
    })
  }, [])

  const [visualVisionEncoder, setVlmVisionEncoder] = useState('google/siglip-base-patch16-224')
  const [visualLLM, setVlmLLM] = useState('Qwen/Qwen2.5-0.5B-Instruct')
  const [visualStage1Epochs, setVlmStage1Epochs] = useState(1)
  const [visualStage2Epochs, setVlmStage2Epochs] = useState(2)

  const [loadingFinetunedModel, setLoadingFinetunedModel] = useState(false)

  const [optimisticJobs, setOptimisticJobs] = useState<TrainingJob[]>([])
  const allJobs = [...optimisticJobs, ...checkpoints.jobs]

  // Clear optimistic jobs when training ends
  useEffect(() => {
    if (session.phase === 'complete' || session.phase === 'error') {
      setOptimisticJobs([])
    }
  }, [session.phase])

  useEffect(() => {
    if (!configLoaded) return
    chatDB.setKV(TRAINING_CONFIG_KEY, {
      method, inputMode, algo, trainingEpochs, trainingLR,
      trainingBatchSize, selectedModel, useLoRA,
    })
  }, [method, inputMode, algo, trainingEpochs, trainingLR, trainingBatchSize, selectedModel, useLoRA, configLoaded])

  useEffect(() => {
    modelController.list().then(models => {
      const ids = models.map(m => m.id)
      setAvailableModels(ids)
      setSelectedModel((prev: string) => prev || ids[0] || '')
    }).catch(() => addToast('Could not load model list — training may be limited', 'info'))
  }, [addToast])

  const canStart = session.trainingRunning ||
    (inputMode === 'dataset' && !datasets.selectedDataset) ||
    (inputMode === 'text' && !textInput.trim()) ||
    (method === 'finetune' && !selectedModel) ||
    (method === 'vlm' && !datasets.selectedDataset)

  const startTraining = useCallback(async (checkpointName?: string) => {
    const hasDataset = inputMode === 'dataset' && datasets.selectedDataset
    const hasText = inputMode === 'text' && textInput.trim()

    if (!hasDataset && !hasText && !checkpointName) {
      addToast('Select a dataset or paste text to train on', 'error'); return
    }

    if (method === 'finetune' && !hasDataset) {
      addToast('Continue training requires a dataset.', 'error'); return
    }

    if (method === 'vlm' && !hasDataset) {
      addToast('Vision model training requires a dataset with image-text pairs', 'error'); return
    }

    const body: Record<string, unknown> = { algo, epochs: trainingEpochs, learning_rate: trainingLR }
    if (trainingBatchSize) body.batch_size = trainingBatchSize
    if (checkpointName) body.checkpoint_name = checkpointName
    if (hasDataset) body.dataset_id = datasets.selectedDataset
    if (hasText) body.source_text = textInput.trim()

    const tempId = `pending-${Date.now()}`
    const now = new Date().toISOString()
    setOptimisticJobs(prev => [...prev, {
      id: tempId, name: `${method} started`, status: 'running',
      progress: 0, created_at: now, status_message: 'Starting...',
    }])

    if (method === 'finetune') {
      session.startFineTune({
        model: selectedModel || 'gpt2',
        dataset: datasets.selectedDataset || 'custom',
        epochs: trainingEpochs,
        batchSize: trainingBatchSize,
        lr: trainingLR,
        useLoRA,
      }, addToast, () => { checkpoints.fetchJobs() })
    } else if (method === 'vlm') {
      session.startVisualTraining({
        dataset: datasets.selectedDataset,
        visionEncoder: visualVisionEncoder,
        llm: visualLLM,
        stage1Epochs: visualStage1Epochs,
        stage2Epochs: visualStage2Epochs,
        useLoRA,
      }, addToast, () => { checkpoints.fetchJobs() })
    } else {
      session.startSSETraining(body, addToast, () => {
        checkpoints.fetchCheckpoints()
      })
    }
  }, [method, inputMode, textInput, algo, trainingEpochs, trainingLR, trainingBatchSize,
      selectedModel, useLoRA, datasets.selectedDataset, visualVisionEncoder, visualLLM,
      visualStage1Epochs, visualStage2Epochs, addToast, session, checkpoints])

  return {
    method, inputMode, textInput, showAdvanced, algo,
    trainingEpochs, trainingLR, trainingBatchSize, availableModels, selectedModel, useLoRA,
    visualVisionEncoder, visualLLM, visualStage1Epochs, visualStage2Epochs, loadingFinetunedModel,
    allJobs,
    setMethod, setInputMode, setTextInput, setShowAdvanced, setAlgo,
    setTrainingEpochs, setTrainingLR, setTrainingBatchSize, setSelectedModel, setUseLoRA,
    setVlmVisionEncoder, setVlmLLM, setVlmStage1Epochs, setVlmStage2Epochs, setLoadingFinetunedModel,
    canStart, startTraining,
  }
}
