// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@/components/training/LossChart', () => ({
  LossChart: () => <div data-testid="loss-chart" />,
}))

vi.mock('@/components/training/TrainingStatus', () => ({
  TrainingErrorBanner: ({ error }: any) => <div data-testid="error-banner">{error}</div>,
}))

import { TrainingPipeline } from './TrainingPipeline'
import type { TrainingFormState } from '@/hooks/useTrainingForm'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { UseTrainingSessionReturn } from '@/hooks/useTrainingSession'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'
import type { Dataset } from '@/lib/dataset-controller'

const ds = (overrides: Partial<Dataset> = {}) => ({
  id: '1',
  name: 'shakespeare',
  type: 'text',
  source: '',
  size: 0,
  created_at: '2026-01-01T00:00:00Z',
  ...overrides,
})

const form: TrainingFormState = {
  method: 'distill',
  inputMode: 'dataset',
  textInput: '',
  showAdvanced: false,
  algo: 'bpe',
  trainingEpochs: 5,
  trainingLR: 1e-3,
  trainingBatchSize: 64,
  availableModels: [],
  selectedModel: '',
  useLoRA: true,
  visualVisionEncoder: '',
  visualLLM: '',
  visualStage1Epochs: 5,
  visualStage2Epochs: 5,
  nativeEmbed: 128,
  nativeLayers: 4,
  nativeHeads: 4,
  nativeBlockSize: 128,
  loadingFinetunedModel: false,
  allJobs: [],
  setMethod: vi.fn(),
  setInputMode: vi.fn(),
  setTextInput: vi.fn(),
  setShowAdvanced: vi.fn(),
  setAlgo: vi.fn(),
  setTrainingEpochs: vi.fn(),
  setTrainingLR: vi.fn(),
  setTrainingBatchSize: vi.fn(),
  setSelectedModel: vi.fn(),
  setUseLoRA: vi.fn(),
  setVlmVisionEncoder: vi.fn(),
  setVlmLLM: vi.fn(),
  setVlmStage1Epochs: vi.fn(),
  setVlmStage2Epochs: vi.fn(),
  setNativeEmbed: vi.fn(),
  setNativeLayers: vi.fn(),
  setNativeHeads: vi.fn(),
  setNativeBlockSize: vi.fn(),
  setLoadingFinetunedModel: vi.fn(),
  applyPreset: vi.fn(),
  customPresets: [],
  saveCustomPreset: vi.fn(),
  deleteCustomPreset: vi.fn(),
  canStart: true,
  startTraining: vi.fn(),
}

const datasets: UseTrainingDatasetsReturn = {
  datasets: [ds({ name: 'shakespeare' })],
  selectedDataset: '1',
  loadingDatasets: false,
  importModalOpen: false,
  datasetPreview: null,
  setSelectedDataset: vi.fn(),
  setImportModalOpen: vi.fn(),
  setDatasetPreview: vi.fn(),
  fetchDatasets: vi.fn(),
}

const session: UseTrainingSessionReturn = {
  phase: 'idle',
  loss: null,
  progress: 0,
  epoch: 0,
  totalEpochs: 0,
  message: '',
  startTime: null,
  lossHistory: [],
  evalResult: null,
  finetunedModelPath: null,
  finetunedModelLoss: null,
  distillCheckpoint: null,
  distillFinalLoss: null,
  distillEpochs: null,
  turboPhase: 'idle',
  turboResult: null,
  turboError: null,
  visualOutputDir: null,
  visualSouPath: null,
  setPhase: vi.fn(),
  setLoss: vi.fn(),
  setProgress: vi.fn(),
  setEpoch: vi.fn(),
  setTotalEpochs: vi.fn(),
  setMessage: vi.fn(),
  setLossHistory: vi.fn(),
  setEvalResult: vi.fn(),
  setFinetunedModelPath: vi.fn(),
  setFinetunedModelLoss: vi.fn(),
  setDistillCheckpoint: vi.fn(),
  setDistillFinalLoss: vi.fn(),
  setDistillEpochs: vi.fn(),
  setTurboPhase: vi.fn(),
  setTurboResult: vi.fn(),
  setTurboError: vi.fn(),
  trainingRunning: false,
  resetTraining: vi.fn(),
  stopTraining: vi.fn(),
  pauseTraining: vi.fn(),
  resumeTraining: vi.fn(),
  paused: false,
  startSSETraining: vi.fn(),
  startFineTune: vi.fn(),
  startVisualTraining: vi.fn(),
  startTurboTrain: vi.fn(),
  turboRunning: false,
}

const checkpoints: UseTrainingCheckpointsReturn = {
  checkpoints: [],
  loadingCheckpoints: false,
  activeCheckpoint: null,
  builds: [],
  loadingBuilds: false,
  jobs: [],
  loadingJobs: false,
  setActiveCheckpoint: vi.fn(),
  setCheckpoints: vi.fn(),
  fetchCheckpoints: vi.fn(),
  fetchBuilds: vi.fn(),
  fetchJobs: vi.fn(),
  handleLoadCheckpoint: vi.fn(),
  handleDeleteCheckpoint: vi.fn(),
}

describe('TrainingPipeline', () => {
  afterEach(cleanup)

  it('renders step indicator with 4 steps', () => {
    render(<TrainingPipeline form={form} datasets={datasets} session={session} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText('Data')).toBeDefined()
    expect(screen.getByText('Configure')).toBeDefined()
    expect(screen.getByText('Train')).toBeDefined()
    expect(screen.getByText('Results')).toBeDefined()
  })

  it('starts on data step', () => {
    render(<TrainingPipeline form={form} datasets={datasets} session={session} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText(/Pick your data/)).toBeDefined()
  })

  it('shows training in progress when session is running', () => {
    const runningSession = { ...session, trainingRunning: true }
    render(<TrainingPipeline form={form} datasets={datasets} session={runningSession} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText('Training in progress')).toBeDefined()
  })

  it('shows error banner on training error', () => {
    const errorSession = { ...session, trainingRunning: true, phase: 'error' as const, message: 'OOM error' }
    render(<TrainingPipeline form={form} datasets={datasets} session={errorSession} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByTestId('error-banner')).toBeDefined()
    expect(screen.getByText('OOM error')).toBeDefined()
  })

  it('shows loss chart during training with data', () => {
    const trainSession = {
      ...session,
      trainingRunning: true,
      lossHistory: [{ step: 1, loss: 0.5 }, { step: 2, loss: 0.3 }],
    }
    render(<TrainingPipeline form={form} datasets={datasets} session={trainSession} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText('Training in progress')).toBeDefined()
  })

  it('shows epoch info during training', () => {
    const trainSession = { ...session, trainingRunning: true, epoch: 3, totalEpochs: 10 }
    render(<TrainingPipeline form={form} datasets={datasets} session={trainSession} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText('Epoch 3/10')).toBeDefined()
  })

  it('shows complete message and Test button', () => {
    const completeSession = { ...session, trainingRunning: true, phase: 'complete' as const }
    render(<TrainingPipeline form={form} datasets={datasets} session={completeSession} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText('Training complete')).toBeDefined()
    expect(screen.getByText('Test model')).toBeDefined()
  })
})
