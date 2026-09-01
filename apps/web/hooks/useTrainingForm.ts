'use client'

import { useCallback, useEffect, useState } from 'react'
import { modelController, trainingJobsController } from '@/lib/controllers'
import type { TrainingJob } from '@/lib/training-controller'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { UseTrainingSessionReturn } from '@/hooks/useTrainingSession'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'
import { chatDB } from '@/lib/db'
import { extractErrorMessage } from '@/lib/error-utils'
import { trackEvent } from '@/lib/dev-log'

export type Method = 'distill' | 'finetune' | 'vlm' | 'native'
export type InputMode = 'dataset' | 'text'

export interface TrainingPreset {
  name: string
  description: string
  method: 'distill' | 'finetune' | 'native' | 'vlm'
  epochs: number
  lr: number
  batchSize: number
  useLoRA?: boolean
  nativeEmbed?: number
  nativeLayers?: number
  nativeHeads?: number
  nativeBlockSize?: number
}

export const BUILT_IN_PRESETS: TrainingPreset[] = [
  { name: 'Quick test', description: 'Fast iteration, minimal training', method: 'distill', epochs: 3, lr: 1e-3, batchSize: 32 },
  { name: 'Personality', description: 'Train character traits from conversations', method: 'distill', epochs: 20, lr: 5e-4, batchSize: 16 },
  { name: 'Fine-tune LoRA', description: 'Adapt an existing model with LoRA', method: 'finetune', epochs: 10, lr: 2e-4, batchSize: 8, useLoRA: true },
  { name: 'Native small', description: 'Tiny transformer from scratch (~150K)', method: 'native', epochs: 100, lr: 3e-4, batchSize: 16, nativeEmbed: 128, nativeLayers: 2, nativeHeads: 4, nativeBlockSize: 128 },
  { name: 'Native large', description: 'Best quality from scratch (~1M)', method: 'native', epochs: 300, lr: 1e-4, batchSize: 8, nativeEmbed: 256, nativeLayers: 4, nativeHeads: 8, nativeBlockSize: 256 },
]

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
  loraRank: number
  loraAlpha: number
  visualVisionEncoder: string
  visualLLM: string
  visualStage1Epochs: number
  visualStage2Epochs: number
  nativeEmbed: number
  nativeLayers: number
  nativeHeads: number
  nativeBlockSize: number
  loadingFinetunedModel: boolean
  resumeCheckpoint: string
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
  setLoraRank: (n: number) => void
  setLoraAlpha: (n: number) => void
  setVlmVisionEncoder: (s: string) => void
  setVlmLLM: (s: string) => void
  setVlmStage1Epochs: (n: number) => void
  setVlmStage2Epochs: (n: number) => void
  setNativeEmbed: (n: number) => void
  setNativeLayers: (n: number) => void
  setNativeHeads: (n: number) => void
  setNativeBlockSize: (n: number) => void
  setLoadingFinetunedModel: (v: boolean) => void
  setResumeCheckpoint: (s: string) => void
  applyPreset: (preset: TrainingPreset) => void
  customPresets: TrainingPreset[]
  saveCustomPreset: (preset: TrainingPreset) => void
  deleteCustomPreset: (name: string) => void
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
  loraRank?: number
  loraAlpha?: number
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
  const [loraRank, setLoraRank] = useState(8)
  const [loraAlpha, setLoraAlpha] = useState(16)
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
        if (saved.loraRank) setLoraRank(saved.loraRank)
        if (saved.loraAlpha) setLoraAlpha(saved.loraAlpha)
      }
      setConfigLoaded(true)
    })
  }, [])

  const [visualVisionEncoder, setVlmVisionEncoder] = useState('google/siglip-base-patch16-224')
  const [visualLLM, setVlmLLM] = useState('Qwen/Qwen2.5-0.5B-Instruct')
  const [visualStage1Epochs, setVlmStage1Epochs] = useState(1)
  const [visualStage2Epochs, setVlmStage2Epochs] = useState(2)

  const [nativeEmbed, setNativeEmbed] = useState(128)
  const [nativeLayers, setNativeLayers] = useState(4)
  const [nativeHeads, setNativeHeads] = useState(4)
  const [nativeBlockSize, setNativeBlockSize] = useState(128)

  const [loadingFinetunedModel, setLoadingFinetunedModel] = useState(false)
  const [resumeCheckpoint, setResumeCheckpoint] = useState('')

  const [customPresets, setCustomPresets] = useState<TrainingPreset[]>([])

  useEffect(() => {
    let cancelled = false
    chatDB.getKV<TrainingPreset[]>('training-presets').then(presets => {
      if (!cancelled && presets) {
        setCustomPresets(presets)
      }
    }).catch(() => {
      // ignore load errors — keep empty presets
    })
    return () => { cancelled = true }
  }, [])

  const saveCustomPreset = useCallback((preset: TrainingPreset) => {
    trackEvent('training_preset_saved', { name: preset.name })
    setCustomPresets(prev => {
      const next = [...prev.filter(p => p.name !== preset.name), preset]
      chatDB.setKV('training-presets', next).catch(() => {})
      return next
    })
  }, [])

  const deleteCustomPreset = useCallback((name: string) => {
    trackEvent('training_preset_deleted', { name })
    setCustomPresets(prev => {
      const next = prev.filter(p => p.name !== name)
      chatDB.setKV('training-presets', next).catch(() => {})
      return next
    })
  }, [])

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

  const applyPreset = useCallback((preset: TrainingPreset) => {
    trackEvent('training_preset_applied', { preset: preset.name })
    setMethod(preset.method)
    setTrainingEpochs(preset.epochs)
    setTrainingLR(preset.lr)
    setTrainingBatchSize(preset.batchSize)
    if (preset.useLoRA !== undefined) setUseLoRA(preset.useLoRA)
    if (preset.nativeEmbed !== undefined) setNativeEmbed(preset.nativeEmbed)
    if (preset.nativeLayers !== undefined) setNativeLayers(preset.nativeLayers)
    if (preset.nativeHeads !== undefined) setNativeHeads(preset.nativeHeads)
    if (preset.nativeBlockSize !== undefined) setNativeBlockSize(preset.nativeBlockSize)
  }, [])

  const canStart = !session.trainingRunning &&
    (inputMode !== 'dataset' || !!datasets.selectedDataset) &&
    (inputMode !== 'text' || !!textInput.trim()) &&
    (method !== 'finetune' || !!selectedModel) &&
    (method !== 'vlm' || !!datasets.selectedDataset)

  const startTraining = useCallback(async (checkpointName?: string) => {
    trackEvent('training_started', { method, input_mode: inputMode })
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

    if (method === 'native' && !hasDataset && !hasText) {
      addToast('Select a dataset or paste text for native training', 'error'); return
    }

    const body: Record<string, unknown> = { algo, epochs: trainingEpochs, learning_rate: trainingLR }
    if (trainingBatchSize) body.batch_size = trainingBatchSize
    if (checkpointName) body.checkpoint_name = checkpointName
    if (hasDataset) body.dataset_id = datasets.selectedDataset
    if (hasText) body.source_text = textInput.trim()

    if (method === 'native') {
      body.n_embed = nativeEmbed
      body.n_layer = nativeLayers
      body.n_head = nativeHeads
      body.block_size = nativeBlockSize
      body.checkpoint_dir = 'models/slonet-native'
    }

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
        loraRank,
        loraAlpha,
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
      trainingJobsController.startAutoTrain(body).then(() => {
        addToast('Training started', 'info')
        checkpoints.fetchCheckpoints()
      }).catch((e: unknown) => {
        addToast(extractErrorMessage(e, 'Could not start training'), 'error')
      })
    }
  }, [method, inputMode, textInput, algo, trainingEpochs, trainingLR, trainingBatchSize,
      selectedModel, useLoRA, datasets.selectedDataset, visualVisionEncoder, visualLLM,
      visualStage1Epochs, visualStage2Epochs, addToast, session, checkpoints])

  return {
    method, inputMode, textInput, showAdvanced, algo,
    trainingEpochs, trainingLR, trainingBatchSize, availableModels, selectedModel, useLoRA,
    loraRank, loraAlpha,
    visualVisionEncoder, visualLLM, visualStage1Epochs, visualStage2Epochs,
    nativeEmbed, nativeLayers, nativeHeads, nativeBlockSize,
    loadingFinetunedModel,
    resumeCheckpoint,
    allJobs,
    setMethod, setInputMode, setTextInput, setShowAdvanced, setAlgo,
    setTrainingEpochs, setTrainingLR, setTrainingBatchSize, setSelectedModel, setUseLoRA,
    setLoraRank, setLoraAlpha,
    setVlmVisionEncoder, setVlmLLM, setVlmStage1Epochs, setVlmStage2Epochs,
    setNativeEmbed, setNativeLayers, setNativeHeads, setNativeBlockSize,
    setLoadingFinetunedModel,
    setResumeCheckpoint,
    applyPreset,
    customPresets, saveCustomPreset, deleteCustomPreset,
    canStart, startTraining,
  }
}
