import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }))
vi.mock('next/dynamic', () => ({
  default: () => {
    const Mock = (props: any) => <div data-testid="loss-chart" />
    Mock.displayName = 'LossChart'
    return Mock
  },
}))
vi.mock('@/components/training/DatasetSelector', () => ({
  DatasetSelector: (props: any) => <div data-testid="dataset-selector" />,
}))
vi.mock('@/components/training/TrainingStatus', () => ({
  TrainingErrorBanner: (props: any) => <div data-testid="error-banner">{props.error}</div>,
}))
vi.mock('@/lib/controllers', () => ({
  modelController: { loadModelPath: vi.fn(), loadVisualModel: vi.fn() },
  trainingJobsController: { loadFineTuned: vi.fn().mockResolvedValue({ status: 'loaded' }) },
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => vi.fn(),
}))

import { TrainingFormCard } from './TrainingFormCard'
import type { TrainingFormState } from '@/hooks/useTrainingForm'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { UseTrainingSessionReturn } from '@/hooks/useTrainingSession'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

function makeForm(overrides: Partial<TrainingFormState> = {}): TrainingFormState {
  return {
    method: 'distill', setMethod: vi.fn(),
    inputMode: 'dataset', setInputMode: vi.fn(),
    textInput: '', setTextInput: vi.fn(),
    trainingEpochs: 10, setTrainingEpochs: vi.fn(),
    trainingLR: 0.001, setTrainingLR: vi.fn(),
    trainingBatchSize: 32, setTrainingBatchSize: vi.fn(),
    selectedModel: '', setSelectedModel: vi.fn(),
    availableModels: [],
    useLoRA: false, setUseLoRA: vi.fn(),
    algo: 'bpe', setAlgo: vi.fn(),
    showAdvanced: false, setShowAdvanced: vi.fn(),
    canStart: false,
    startTraining: vi.fn(),
    loadingFinetunedModel: false, setLoadingFinetunedModel: vi.fn(),
    visualVisionEncoder: 'google/siglip-base-patch16-224', setVlmVisionEncoder: vi.fn(),
    visualLLM: '', setVlmLLM: vi.fn(),
    visualStage1Epochs: 3, setVlmStage1Epochs: vi.fn(),
    visualStage2Epochs: 10, setVlmStage2Epochs: vi.fn(),
    ...overrides,
  } as TrainingFormState
}

function makeDatasets(overrides: Partial<UseTrainingDatasetsReturn> = {}): UseTrainingDatasetsReturn {
  return {
    datasets: [], datasetsLoading: false,
    selectedDataset: null, setSelectedDataset: vi.fn(),
    datasetPreview: null, fetchPreview: vi.fn(),
    importDataset: vi.fn(),
    ...overrides,
  } as UseTrainingDatasetsReturn
}

function makeSession(overrides: Partial<UseTrainingSessionReturn> = {}): UseTrainingSessionReturn {
  return {
    phase: 'idle', trainingRunning: false, message: '',
    progress: 0, epoch: 0, totalEpochs: 0, loss: null,
    lossHistory: [], startTime: null, paused: false,
    finetunedModelPath: null, finetunedModelLoss: null,
    distillCheckpoint: null, distillFinalLoss: null, distillEpochs: null,
    evalResult: null, visualOutputDir: null,
    startTraining: vi.fn(), stopTraining: vi.fn(),
    pauseTraining: vi.fn(), resumeTraining: vi.fn(),
    resetTraining: vi.fn(),
    ...overrides,
  } as UseTrainingSessionReturn
}

function makeCheckpoints(overrides: Partial<UseTrainingCheckpointsReturn> = {}): UseTrainingCheckpointsReturn {
  return {
    checkpoints: [], checkpointsLoading: false,
    handleLoadCheckpoint: vi.fn(), handleDeleteCheckpoint: vi.fn(),
    ...overrides,
  } as UseTrainingCheckpointsReturn
}

function renderCard(formOverrides = {}, sessionOverrides = {}, datasetOverrides = {}) {
  const onTest = vi.fn()
  return {
    onTest,
    ...render(
      <TrainingFormCard
        form={makeForm(formOverrides)}
        datasets={makeDatasets(datasetOverrides)}
        session={makeSession(sessionOverrides)}
        checkpoints={makeCheckpoints()}
        onTest={onTest}
      />
    ),
  }
}

describe('TrainingFormCard', () => {
  afterEach(cleanup)

  it('renders card title', () => {
    renderCard()
    expect(screen.getByText('Train')).toBeDefined()
  })

  it('shows dataset selector when idle', () => {
    renderCard()
    expect(screen.getByTestId('dataset-selector')).toBeDefined()
  })

  it('hides dataset selector when training running', () => {
    renderCard({}, { trainingRunning: true })
    expect(screen.queryByTestId('dataset-selector')).toBeNull()
  })

  it('shows training progress when running', () => {
    renderCard({}, { trainingRunning: true, progress: 42, epoch: 3, totalEpochs: 10, loss: 0.5 })
    expect(screen.getByText('Training in progress')).toBeDefined()
    expect(screen.getByText('Epoch 3 of 10')).toBeDefined()
    expect(screen.getByText(/Loss: 0\.5000/)).toBeDefined()
  })

  it('shows progress bar width', () => {
    const { container } = renderCard({}, { trainingRunning: true, progress: 60 })
    const bar = container.querySelector('[style*="width: 60%"]')
    expect(bar).toBeTruthy()
  })

  it('shows pause button when not paused', () => {
    renderCard({}, { trainingRunning: true, paused: false })
    expect(screen.getByText('Pause')).toBeDefined()
  })

  it('shows resume button when paused', () => {
    renderCard({}, { trainingRunning: true, paused: true })
    expect(screen.getByText('Resume')).toBeDefined()
    expect(screen.getByText('Training paused')).toBeDefined()
  })

  it('shows stop button during training', () => {
    renderCard({}, { trainingRunning: true })
    expect(screen.getByText('Stop')).toBeDefined()
  })

  it('calls onTest when "Test the model" clicked', () => {
    const { onTest } = renderCard({}, { phase: 'complete', trainingRunning: false })
    fireEvent.click(screen.getByText('Test the model'))
    expect(onTest).toHaveBeenCalled()
  })

  it('shows completion banner with finetuned model', () => {
    renderCard({}, {
      phase: 'complete', trainingRunning: false,
      finetunedModelPath: '/models/my-finetuned',
      finetunedModelLoss: 0.32,
    })
    expect(screen.getByText('Training complete!')).toBeDefined()
    expect(screen.getByText('/models/my-finetuned')).toBeDefined()
    expect(screen.getByText(/0\.3200/)).toBeDefined()
  })

  it('routes "Load model for chat" through the fine-tuned load endpoint', async () => {
    const loadFineTuned = (await import('@/lib/controllers')).trainingJobsController.loadFineTuned as ReturnType<typeof vi.fn>
    renderCard({}, {
      phase: 'complete', trainingRunning: false,
      finetunedModelPath: '/repos/sloughGPT/models/hf-finetuned/gpt2__dataset_1',
    })
    fireEvent.click(screen.getByText('Load model for chat'))
    expect(loadFineTuned).toHaveBeenCalledWith('gpt2__dataset_1')
  })

  it('shows completion banner with distill checkpoint', () => {
    renderCard({}, {
      phase: 'complete', trainingRunning: false,
      distillCheckpoint: 'distill-v1',
      distillFinalLoss: 0.45,
      distillEpochs: 5,
    })
    expect(screen.getByText('Training complete!')).toBeDefined()
    expect(screen.getByText('distill-v1')).toBeDefined()
  })

  it('shows eval result in completion', () => {
    renderCard({}, {
      phase: 'complete', trainingRunning: false,
      evalResult: 'BLEU: 0.85, Perplexity: 12.3',
    })
    expect(screen.getByText(/BLEU: 0\.85/)).toBeDefined()
  })

  it('shows error banner on error phase', () => {
    renderCard({}, { phase: 'error', trainingRunning: false, message: 'OOM killed' })
    expect(screen.getByTestId('error-banner')).toBeDefined()
    expect(screen.getByText('OOM killed')).toBeDefined()
  })

  it('shows "Try in chat" button on completion', () => {
    renderCard({}, { phase: 'complete', trainingRunning: false })
    expect(screen.getByText('Try in chat')).toBeDefined()
  })

  it('shows "Train another" button on completion', () => {
    renderCard({}, { phase: 'complete', trainingRunning: false })
    expect(screen.getByText('Train another')).toBeDefined()
  })

  it('shows advanced toggle', () => {
    renderCard()
    expect(screen.getByText('Show advanced settings')).toBeDefined()
  })

  it('expands advanced settings on click', () => {
    renderCard({ showAdvanced: true })
    expect(screen.getByText('Hide advanced settings')).toBeDefined()
  })

  it('shows method toggles in advanced', () => {
    renderCard({ showAdvanced: true })
    expect(screen.getByText('Train from scratch')).toBeDefined()
    expect(screen.getByText('Continue training')).toBeDefined()
    expect(screen.getByText('Vision model')).toBeDefined()
  })

  it('shows distill description when distill selected', () => {
    renderCard({ showAdvanced: true, method: 'distill' })
    expect(screen.getByText('Train a small model from text data — no teacher needed')).toBeDefined()
  })

  it('shows finetune description', () => {
    renderCard({ showAdvanced: true, method: 'finetune' })
    expect(screen.getByText('Continue training an existing model on new data')).toBeDefined()
  })

  it('shows VLM description', () => {
    renderCard({ showAdvanced: true, method: 'vlm' })
    expect(screen.getByText('Teach the AI to understand images and text')).toBeDefined()
  })

  it('shows text input when input mode is text', () => {
    renderCard({ showAdvanced: true, inputMode: 'text', method: 'distill' })
    expect(screen.getByLabelText('Training text input')).toBeDefined()
  })

  it('hides text input when input mode is dataset', () => {
    renderCard({ showAdvanced: true, inputMode: 'dataset', method: 'distill' })
    expect(screen.queryByLabelText('Training text input')).toBeNull()
  })

  it('shows epochs/LR/batch inputs in advanced', () => {
    renderCard({ showAdvanced: true })
    expect(screen.getByText('Epochs')).toBeDefined()
    expect(screen.getByText('LR')).toBeDefined()
    expect(screen.getByText('Batch')).toBeDefined()
  })

  it('shows dataset preview when available', () => {
    renderCard({}, {}, {
      datasetPreview: { dataset_id: 'test', samples: [{ content: 'Hello world', path: '', language: 'en', size: 11 }, { content: 'Test line', path: '', language: 'en', size: 9 }], total_samples: 2, total_chars: 20, languages: { en: 2 } },
    })
    expect(screen.getByText('Dataset preview')).toBeDefined()
    expect(screen.getByText('Hello world')).toBeDefined()
  })

  it('calls resetTraining when "Train another" clicked', () => {
    const resetTraining = vi.fn()
    renderCard({}, { phase: 'complete', trainingRunning: false, resetTraining })
    fireEvent.click(screen.getByText('Train another'))
    expect(resetTraining).toHaveBeenCalled()
  })

  it('calls stopTraining when "Stop" clicked', () => {
    const stopTraining = vi.fn()
    renderCard({}, { trainingRunning: true, stopTraining })
    fireEvent.click(screen.getByText('Stop'))
    expect(stopTraining).toHaveBeenCalled()
  })

  it('calls pauseTraining when "Pause" clicked', () => {
    const pauseTraining = vi.fn()
    renderCard({}, { trainingRunning: true, paused: false, pauseTraining })
    fireEvent.click(screen.getByText('Pause'))
    expect(pauseTraining).toHaveBeenCalled()
  })

  it('calls resumeTraining when "Resume" clicked', () => {
    const resumeTraining = vi.fn()
    renderCard({}, { trainingRunning: true, paused: true, resumeTraining })
    fireEvent.click(screen.getByText('Resume'))
    expect(resumeTraining).toHaveBeenCalled()
  })
})
