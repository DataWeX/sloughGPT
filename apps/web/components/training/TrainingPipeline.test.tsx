// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  const button = ({ children, onClick, disabled, variant, size, className }: any) => (
    <button onClick={onClick} disabled={disabled} data-variant={variant} data-size={size} className={className}>{children}</button>
  )
  return {
    Card: passthrough,
    CardContent: passthrough,
    CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div data-testid="card-title">{children}</div>,
    Button: button,
    ToggleGroup: ({ value, onValueChange, children }: any) => (
      <div data-testid="toggle-group" data-value={value}>{children}</div>
    ),
    ToggleGroupItem: ({ value, children, ...props }: any) => (
      <button data-value={value} onClick={() => props.onClick?.()}>{children}</button>
    ),
    Select: ({ value, onValueChange, children }: any) => (
      <div data-testid="select" data-value={value}>{children}</div>
    ),
    SelectTrigger: ({ children, ...props }: any) => <div>{children}</div>,
    SelectValue: ({ placeholder }: any) => <span>{placeholder}</span>,
    SelectContent: passthrough,
    SelectItem: ({ value, children }: any) => <div data-value={value}>{children}</div>,
  }
})

vi.mock('next/dynamic', () => {
  const dynamic = () => {
    const MockChart = (props: any) => <div data-testid="loss-chart">{JSON.stringify(props.data?.length ?? 0)} points</div>
    MockChart.displayName = 'LossChart'
    return MockChart
  }
  return { __esModule: true, default: dynamic }
})

vi.mock('@/components/training/DatasetSelector', () => ({
  DatasetSelector: ({ value, onChange }: any) => (
    <div data-testid="dataset-selector">
      <button onClick={() => onChange('test-ds')}>Select test-ds</button>
      {value && <span>selected: {value}</span>}
    </div>
  ),
}))

vi.mock('@/components/training/TrainingStatus', () => ({
  TrainingErrorBanner: ({ error, onRetry, onDismiss }: any) => (
    <div data-testid="error-banner">
      <span>{error}</span>
      <button onClick={onRetry}>retry</button>
      <button onClick={onDismiss}>dismiss</button>
    </div>
  ),
}))

vi.mock('@/components/training/TrainingPresets', () => ({
  TrainingPresets: () => <div data-testid="training-presets" />,
}))

function makeForm(overrides: Record<string, any> = {}) {
  return {
    method: 'distill' as const,
    inputMode: 'dataset' as const,
    textInput: '',
    showAdvanced: false,
    algo: '',
    trainingEpochs: 3,
    trainingLR: 0.001,
    trainingBatchSize: 8,
    availableModels: [],
    selectedModel: '',
    useLoRA: false,
    visualVisionEncoder: '',
    visualLLM: '',
    visualStage1Epochs: 1,
    visualStage2Epochs: 2,
    nativeEmbed: 64,
    nativeLayers: 4,
    nativeHeads: 4,
    nativeBlockSize: 128,
    loadingFinetunedModel: false,
    allJobs: [] as any[],
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
    customPresets: [] as any[],
    saveCustomPreset: vi.fn(),
    deleteCustomPreset: vi.fn(),
    canStart: true,
    startTraining: vi.fn(),
    ...overrides,
  }
}

function makeDatasets(overrides: Record<string, any> = {}) {
  return {
    datasets: [{ id: 'ds-1', name: 'Dataset 1' }] as any[],
    selectedDataset: '',
    loadingDatasets: false,
    importModalOpen: false,
    datasetPreview: null as any,
    setSelectedDataset: vi.fn(),
    setImportModalOpen: vi.fn(),
    setDatasetPreview: vi.fn(),
    fetchDatasets: vi.fn(),
    ...overrides,
  }
}

function makeSession(overrides: Record<string, any> = {}) {
  return {
    trainingRunning: false,
    epoch: 0,
    totalEpochs: 0,
    loss: null as number | null,
    lossHistory: [] as any[],
    phase: 'idle' as string,
    distillCheckpoint: null as string | null,
    message: '',
    resetTraining: vi.fn(),
    setPhase: vi.fn(),
    setLoss: vi.fn(),
    setProgress: vi.fn(),
    setEpoch: vi.fn(),
    setTotalEpochs: vi.fn(),
    setLossHistory: vi.fn(),
    setDistillCheckpoint: vi.fn(),
    setMessage: vi.fn(),
    setTrainingRunning: vi.fn(),
    setEvalResult: vi.fn(),
    setFinetunedModelPath: vi.fn(),
    setFinetunedModelLoss: vi.fn(),
    setDistillFinalLoss: vi.fn(),
    setDistillEpochs: vi.fn(),
    setTurboPhase: vi.fn(),
    setTurboResult: vi.fn(),
    setTurboError: vi.fn(),
    setStartTime: vi.fn(),
    setVisualOutputDir: vi.fn(),
    setVisualSouPath: vi.fn(),
    progress: 0,
    startTime: null,
    evalResult: null,
    finetunedModelPath: null,
    finetunedModelLoss: null,
    distillFinalLoss: null,
    distillEpochs: null,
    turboPhase: 'idle' as 'idle' | 'training' | 'complete' | 'error',
    turboResult: null,
    turboError: null,
    visualOutputDir: null,
    visualSouPath: null,
    paused: false,
    stopTraining: vi.fn(),
    pauseTraining: vi.fn(),
    resumeTraining: vi.fn(),
    startSSETraining: vi.fn(),
    startFineTune: vi.fn(),
    startVisualTraining: vi.fn(),
    startTurboTrain: vi.fn(),
    turboRunning: false,
    ...overrides,
  }
}

function makeCheckpoints(overrides: Record<string, any> = {}) {
  return {
    checkpoints: [] as any[],
    loadingCheckpoints: false,
    loadingJobs: false,
    activeCheckpoint: null,
    builds: [],
    loadingBuilds: false,
    jobs: [],
    setActiveCheckpoint: vi.fn(),
    setCheckpoints: vi.fn(),
    fetchCheckpoints: vi.fn(),
    fetchBuilds: vi.fn(),
    fetchJobs: vi.fn(),
    handleLoadCheckpoint: vi.fn(),
    handleDeleteCheckpoint: vi.fn(),
    ...overrides,
  }
}

afterEach(() => { cleanup() })

describe('TrainingPipeline', () => {
  let TrainingPipeline: typeof import('./TrainingPipeline').TrainingPipeline

  beforeEach(async () => {
    vi.clearAllMocks()
    const mod = await import('./TrainingPipeline')
    TrainingPipeline = mod.TrainingPipeline
  })

  it('renders 4 step indicators', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.getAllByTestId('card-title')).toHaveLength(1)
    expect(screen.getByText('Data')).toBeTruthy()
    expect(screen.getByText('Configure')).toBeTruthy()
    expect(screen.getByText('Train')).toBeTruthy()
    expect(screen.getByText('Results')).toBeTruthy()
  })

  it('starts on the Data step', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.getByText('1. Pick your data')).toBeTruthy()
    expect(screen.getByTestId('dataset-selector')).toBeTruthy()
  })

  it('disables Next when no dataset selected and inputMode is dataset', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    const nextBtn = screen.getByText('Next: Configure')
    expect(nextBtn.closest('button')!.disabled).toBe(true)
  })

  it('enables Next when a dataset is selected', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    const nextBtn = screen.getByText('Next: Configure')
    expect(nextBtn.closest('button')!.disabled).toBe(false)
  })

  it('enables Next when inputMode is text and textInput is non-empty', () => {
    render(<TrainingPipeline form={makeForm({ inputMode: 'text', textInput: 'hello world' })} datasets={makeDatasets()} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    const nextBtn = screen.getByText('Next: Configure')
    expect(nextBtn.closest('button')!.disabled).toBe(false)
  })

  it('shows dataset preview when available', () => {
    const preview = { samples: [{ content: 'sample 1' }, { content: 'sample 2' }], total_samples: 100, total_chars: 5000 }
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets({ datasetPreview: preview })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.getByText('Preview')).toBeTruthy()
    expect(screen.getByText('100')).toBeTruthy()
    expect(screen.getByText('5,000')).toBeTruthy()
    expect(screen.getByText('sample 1')).toBeTruthy()
  })

  it('shows helper text when no dataset selected', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.getByText(/Select a dataset or switch to paste text/)).toBeTruthy()
  })

  it('advances to Configure step on Next click', async () => {
    render(<TrainingPipeline form={makeForm({ canStart: true })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    expect(screen.getByText('2. Configure training')).toBeTruthy()
  })

  it('shows mode toggles (Text / Vision)', async () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    expect(screen.getByText('Text')).toBeTruthy()
    expect(screen.getByText('Vision')).toBeTruthy()
  })

  it('shows text sub-methods in Text mode', async () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    expect(screen.getByText('Train from scratch')).toBeTruthy()
    expect(screen.getByText('Continue training')).toBeTruthy()
    expect(screen.getByText('Native SloNet')).toBeTruthy()
  })

  it('hides sub-methods in Vision mode', async () => {
    render(<TrainingPipeline form={makeForm({ method: 'vlm' })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    expect(screen.queryByText('Train from scratch')).toBeFalsy()
    expect(screen.queryByText('Continue training')).toBeFalsy()
    expect(screen.queryByText('Native SloNet')).toBeFalsy()
  })

  it('shows data source toggles for non-VLM methods', async () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    expect(screen.getByText('Use a dataset')).toBeTruthy()
    expect(screen.getByText('Paste text')).toBeTruthy()
  })

  it('hides data source toggles for VLM method', async () => {
    render(<TrainingPipeline form={makeForm({ method: 'vlm' })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    expect(screen.queryByText('Use a dataset')).toBeFalsy()
    expect(screen.queryByText('Paste text')).toBeFalsy()
  })

  it('shows text input when inputMode is text', async () => {
    render(<TrainingPipeline form={makeForm({ inputMode: 'text' })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    expect(screen.getByLabelText('Training text input')).toBeTruthy()
  })

  it('hides text input when inputMode is dataset', async () => {
    render(<TrainingPipeline form={makeForm({ inputMode: 'dataset' })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    expect(screen.queryByLabelText('Training text input')).toBeFalsy()
  })

  it('shows hyperparameter inputs', async () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    expect(screen.getByText('Epochs')).toBeTruthy()
    expect(screen.getByText('Batch size')).toBeTruthy()
    expect(screen.getByText('Learning rate')).toBeTruthy()
  })

  it('shows LoRA checkbox for finetune method', async () => {
    render(<TrainingPipeline form={makeForm({ method: 'finetune' })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    expect(screen.getByText(/Use LoRA/)).toBeTruthy()
  })

  it('hides LoRA checkbox for distill method', async () => {
    render(<TrainingPipeline form={makeForm({ method: 'distill' })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    expect(screen.queryByText(/Use LoRA/)).toBeFalsy()
  })

  it('disables Next when canStart is false', async () => {
    render(<TrainingPipeline form={makeForm({ canStart: false })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    const nextBtn = screen.getByText('Next: Train')
    expect(nextBtn.closest('button')!.disabled).toBe(true)
  })

  it('disables Next when epochs is out of range', async () => {
    render(<TrainingPipeline form={makeForm({ trainingEpochs: 0 })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    expect(screen.getByText('Epochs must be 1\u2013500')).toBeTruthy()
    const nextBtn = screen.getByText('Next: Train')
    expect(nextBtn.closest('button')!.disabled).toBe(true)
  })

  it('disables Next when batch size is out of range', async () => {
    render(<TrainingPipeline form={makeForm({ trainingBatchSize: 300 })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    expect(screen.getByText('Batch size must be 1\u2013256')).toBeTruthy()
    const nextBtn = screen.getByText('Next: Train')
    expect(nextBtn.closest('button')!.disabled).toBe(true)
  })

  it('disables Next when learning rate is out of range', async () => {
    render(<TrainingPipeline form={makeForm({ trainingLR: 0 })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    expect(screen.getByText('Learning rate must be 0\u20131')).toBeTruthy()
    const nextBtn = screen.getByText('Next: Train')
    expect(nextBtn.closest('button')!.disabled).toBe(true)
  })

  it('enables Next when all hyperparams are valid', async () => {
    render(<TrainingPipeline form={makeForm({ trainingEpochs: 5, trainingBatchSize: 16, trainingLR: 0.001 })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    const nextBtn = screen.getByText('Next: Train')
    expect(nextBtn.closest('button')!.disabled).toBe(false)
  })

  it('goes back to Data step on Back click', async () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    await act(async () => { screen.getByText('Back').closest('button')!.click() })
    expect(screen.getByText('1. Pick your data')).toBeTruthy()
  })

  it('advances to Train step and shows config summary', async () => {
    render(<TrainingPipeline form={makeForm({ trainingEpochs: 5, trainingBatchSize: 16, trainingLR: 0.0005 })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    await act(async () => { screen.getByText('Next: Train').closest('button')!.click() })
    expect(screen.getByText('3. Train')).toBeTruthy()
    expect(screen.getByText('distill')).toBeTruthy()
    expect(screen.getByText('ds-1')).toBeTruthy()
  })

  it('calls startTraining when Start button is clicked', async () => {
    const startTraining = vi.fn()
    render(<TrainingPipeline form={makeForm({ canStart: true, startTraining })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    await act(async () => { screen.getByText('Next: Train').closest('button')!.click() })
    await act(async () => { screen.getByText('Start training').closest('button')!.click() })
    expect(startTraining).toHaveBeenCalled()
  })

  it('shows "Train on pasted text" for distill+text mode', async () => {
    render(<TrainingPipeline form={makeForm({ method: 'distill', inputMode: 'text', textInput: 'some text' })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    await act(async () => { screen.getByText('Next: Train').closest('button')!.click() })
    expect(screen.getByText('Train on pasted text')).toBeTruthy()
  })

  it('shows "Start vision training" for VLM method', async () => {
    render(<TrainingPipeline form={makeForm({ method: 'vlm' })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    await act(async () => { screen.getByText('Next: Train').closest('button')!.click() })
    expect(screen.getByText('Start vision training')).toBeTruthy()
  })

  it('goes back to Configure step on Back click', async () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets({ selectedDataset: 'ds-1' })} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    await act(async () => { screen.getByText('Next: Configure').closest('button')!.click() })
    await act(async () => { screen.getByText('Next: Train').closest('button')!.click() })
    await act(async () => { screen.getByText('Back').closest('button')!.click() })
    expect(screen.getByText('2. Configure training')).toBeTruthy()
  })

  it('shows training card when training is running', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession({ trainingRunning: true })} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.getByText('Training in progress')).toBeTruthy()
  })

  it('shows epoch counter during training', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession({ trainingRunning: true, epoch: 3, totalEpochs: 10 })} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.getByText('Epoch 3/10')).toBeTruthy()
  })

  it('shows loss during training', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession({ trainingRunning: true, loss: 1.2345 })} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.getByText('Loss: 1.2345')).toBeTruthy()
  })

  it('shows loss chart when lossHistory has data', () => {
    const history = [{ step: 1, loss: 2.0 }, { step: 2, loss: 1.5 }]
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession({ trainingRunning: true, lossHistory: history })} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.getByTestId('loss-chart')).toBeTruthy()
  })

  it('hides loss chart when lossHistory is empty', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession({ trainingRunning: true })} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.queryByTestId('loss-chart')).toBeFalsy()
  })

  it('shows completion banner when phase is complete', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession({ trainingRunning: true, phase: 'complete' })} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.getByText('Training complete')).toBeTruthy()
  })

  it('shows checkpoint name in completion banner', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession({ trainingRunning: true, phase: 'complete', distillCheckpoint: 'cp-1' })} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.getByText(/cp-1/)).toBeTruthy()
  })

  it('shows Test model button on completion', () => {
    const onTest = vi.fn()
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession({ trainingRunning: true, phase: 'complete' })} checkpoints={makeCheckpoints()} onTest={onTest} />)
    expect(screen.getByText('Test model')).toBeTruthy()
  })

  it('shows Load checkpoint button when distillCheckpoint exists', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession({ trainingRunning: true, phase: 'complete', distillCheckpoint: 'cp-1' })} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.getByText('Load checkpoint')).toBeTruthy()
  })

  it('hides Load checkpoint button when no distillCheckpoint', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession({ trainingRunning: true, phase: 'complete' })} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.queryByText('Load checkpoint')).toBeFalsy()
  })

  it('shows error banner when phase is error', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession({ trainingRunning: true, phase: 'error', message: 'Out of memory' })} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.getByTestId('error-banner')).toBeTruthy()
    expect(screen.getByText('Out of memory')).toBeTruthy()
  })

  it('shows default error message when no message', () => {
    render(<TrainingPipeline form={makeForm()} datasets={makeDatasets()} session={makeSession({ trainingRunning: true, phase: 'error' })} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.getByText('Training failed')).toBeTruthy()
  })

  it('shows training card when a running job exists', () => {
    const form = makeForm({ allJobs: [{ id: 'j1', status: 'running' }] })
    render(<TrainingPipeline form={form} datasets={makeDatasets()} session={makeSession()} checkpoints={makeCheckpoints()} onTest={vi.fn()} />)
    expect(screen.getByText('Training in progress')).toBeTruthy()
  })
})
