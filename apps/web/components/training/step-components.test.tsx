// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

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

vi.mock('@/components/training/DatasetSelector', () => ({
  DatasetSelector: ({ value, onChange }: any) => (
    <div data-testid="dataset-selector">
      <button onClick={() => onChange('test-ds')}>Select test-ds</button>
      {value && <span>selected: {value}</span>}
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
    customPresets: [],
    canStart: true,
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
    saveCustomPreset: vi.fn(),
    deleteCustomPreset: vi.fn(),
    startTraining: vi.fn(),
    ...overrides,
  }
}

function makeDatasets(overrides: Record<string, any> = {}) {
  return {
    datasets: [],
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

describe('DataStep', () => {
  let DataStep: typeof import('./DataStep').DataStep

  beforeEach(async () => {
    vi.clearAllMocks()
    const mod = await import('./DataStep')
    DataStep = mod.DataStep
  })

  it('renders card title', () => {
    render(<DataStep form={makeForm()} datasets={makeDatasets()} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('1. Pick your data')).toBeTruthy()
  })

  it('renders dataset selector', () => {
    render(<DataStep form={makeForm()} datasets={makeDatasets()} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByTestId('dataset-selector')).toBeTruthy()
  })

  it('disables Next when no dataset and inputMode is dataset', () => {
    render(<DataStep form={makeForm()} datasets={makeDatasets()} onNext={vi.fn()} onBack={vi.fn()} />)
    const nextBtn = screen.getByText('Next: Configure')
    expect(nextBtn.closest('button')!.disabled).toBe(true)
  })

  it('enables Next when dataset is selected', () => {
    render(<DataStep form={makeForm()} datasets={makeDatasets({ selectedDataset: 'ds-1' })} onNext={vi.fn()} onBack={vi.fn()} />)
    const nextBtn = screen.getByText('Next: Configure')
    expect(nextBtn.closest('button')!.disabled).toBe(false)
  })

  it('enables Next when text input is non-empty', () => {
    render(<DataStep form={makeForm({ inputMode: 'text', textInput: 'hello' })} datasets={makeDatasets()} onNext={vi.fn()} onBack={vi.fn()} />)
    const nextBtn = screen.getByText('Next: Configure')
    expect(nextBtn.closest('button')!.disabled).toBe(false)
  })

  it('calls onNext when Next is clicked', async () => {
    const onNext = vi.fn()
    render(<DataStep form={makeForm()} datasets={makeDatasets({ selectedDataset: 'ds-1' })} onNext={onNext} onBack={vi.fn()} />)
    screen.getByText('Next: Configure').closest('button')!.click()
    expect(onNext).toHaveBeenCalled()
  })

  it('shows preview when datasetPreview is provided', () => {
    const preview = { samples: [{ content: 'sample 1' }], total_samples: 50, total_chars: 2500 }
    render(<DataStep form={makeForm()} datasets={makeDatasets({ datasetPreview: preview })} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Preview')).toBeTruthy()
    expect(screen.getByText('50')).toBeTruthy()
  })

  it('shows helper text when no dataset selected', () => {
    render(<DataStep form={makeForm()} datasets={makeDatasets()} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText(/Select a dataset or switch to paste text/)).toBeTruthy()
  })
})

describe('TrainStep', () => {
  let TrainStep: typeof import('./TrainStep').TrainStep

  beforeEach(async () => {
    vi.clearAllMocks()
    const mod = await import('./TrainStep')
    TrainStep = mod.TrainStep
  })

  it('renders card title', () => {
    render(<TrainStep form={makeForm()} datasets={makeDatasets()} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('3. Train')).toBeTruthy()
  })

  it('shows config summary', () => {
    render(<TrainStep form={makeForm({ trainingEpochs: 5, trainingBatchSize: 16 })} datasets={makeDatasets({ selectedDataset: 'ds-1' })} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('distill')).toBeTruthy()
    expect(screen.getByText('5')).toBeTruthy()
    expect(screen.getByText('16')).toBeTruthy()
  })

  it('disables Start when canStart is false', () => {
    render(<TrainStep form={makeForm({ canStart: false })} datasets={makeDatasets()} onNext={vi.fn()} onBack={vi.fn()} />)
    const startBtn = screen.getByText('Start training')
    expect(startBtn.closest('button')!.disabled).toBe(true)
  })

  it('calls startTraining when Start is clicked', async () => {
    const startTraining = vi.fn()
    render(<TrainStep form={makeForm({ canStart: true, startTraining })} datasets={makeDatasets()} onNext={vi.fn()} onBack={vi.fn()} />)
    screen.getByText('Start training').closest('button')!.click()
    expect(startTraining).toHaveBeenCalled()
  })

  it('calls onBack when Back is clicked', () => {
    const onBack = vi.fn()
    render(<TrainStep form={makeForm()} datasets={makeDatasets()} onNext={vi.fn()} onBack={onBack} />)
    screen.getByText('Back').closest('button')!.click()
    expect(onBack).toHaveBeenCalled()
  })

  it('shows "Train on pasted text" for distill+text', () => {
    render(<TrainStep form={makeForm({ method: 'distill', inputMode: 'text', textInput: 'some text' })} datasets={makeDatasets()} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Train on pasted text')).toBeTruthy()
  })

  it('shows "Start vision training" for VLM', () => {
    render(<TrainStep form={makeForm({ method: 'vlm' })} datasets={makeDatasets()} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Start vision training')).toBeTruthy()
  })
})

describe('ResultsStep', () => {
  let ResultsStep: typeof import('./ResultsStep').ResultsStep

  beforeEach(async () => {
    vi.clearAllMocks()
    const mod = await import('./ResultsStep')
    ResultsStep = mod.ResultsStep
  })

  it('renders card title', () => {
    render(<ResultsStep checkpoints={makeCheckpoints()} goToTrain={vi.fn()} />)
    expect(screen.getByText('4. Results')).toBeTruthy()
  })

  it('shows empty state when no checkpoints', () => {
    render(<ResultsStep checkpoints={makeCheckpoints()} goToTrain={vi.fn()} />)
    expect(screen.getByText(/No checkpoints yet/)).toBeTruthy()
  })

  it('shows checkpoints when available', () => {
    const checkpoints = makeCheckpoints({
      checkpoints: [{ name: 'cp-1', loss: 0.5, tags: ['test'] }],
    })
    render(<ResultsStep checkpoints={checkpoints} goToTrain={vi.fn()} />)
    expect(screen.getByText('cp-1')).toBeTruthy()
    expect(screen.getByText('Loss: 0.5000')).toBeTruthy()
    expect(screen.getByText('Tags: test')).toBeTruthy()
  })

  it('calls handleLoadCheckpoint when Load is clicked', () => {
    const handleLoadCheckpoint = vi.fn()
    const checkpoints = makeCheckpoints({
      checkpoints: [{ name: 'cp-1', loss: 0.5 }],
      handleLoadCheckpoint,
    })
    render(<ResultsStep checkpoints={checkpoints} goToTrain={vi.fn()} />)
    screen.getByText('Load').closest('button')!.click()
    expect(handleLoadCheckpoint).toHaveBeenCalledWith('cp-1', expect.any(Function))
  })

  it('calls goToTrain when Train more is clicked', () => {
    const goToTrain = vi.fn()
    render(<ResultsStep checkpoints={makeCheckpoints()} goToTrain={goToTrain} />)
    screen.getByText('Train more').closest('button')!.click()
    expect(goToTrain).toHaveBeenCalled()
  })

  it('shows checkpoint count', () => {
    const checkpoints = makeCheckpoints({
      checkpoints: [
        { name: 'cp-1', loss: 0.5 },
        { name: 'cp-2', loss: 0.3 },
      ],
    })
    render(<ResultsStep checkpoints={checkpoints} goToTrain={vi.fn()} />)
    expect(screen.getByText('2 checkpoint(s) saved')).toBeTruthy()
  })
})
