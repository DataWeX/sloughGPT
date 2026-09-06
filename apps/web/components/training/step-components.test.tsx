// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { makeForm, makeDatasets, makeCheckpoints } from './__test-helper'

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
    Badge: ({ children, variant, size }: any) => <span data-variant={variant} data-size={size}>{children}</span>,
    Checkbox: ({ checked, onCheckedChange, className, ...props }: any) => (
      <input type="checkbox" checked={checked} onChange={() => onCheckedChange?.(!checked)} className={className} {...props} />
    ),
    EmptyCard: ({ title, description, icon }: any) => <div data-testid="empty-card"><div>{title}</div><div>{description}</div></div>,
    SectionHeader: ({ title }: any) => <div data-testid="section-header">{title}</div>,
    FoldSection: ({ heading, children }: any) => <details open><summary>{heading}</summary><div>{children}</div></details>,
    SearchInput: ({ value, onChange, placeholder }: any) => <input value={value} onChange={onChange} placeholder={placeholder} />,
    Spinner: () => <div data-testid="spinner" />,
    IconFile: () => <span data-testid="icon-file" />,
    IconUpload: () => <span data-testid="icon-upload" />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children, asChild }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <div>{children}</div>,
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

  it('enables Next when no dataset so user can reach paste-text step', () => {
    render(<DataStep form={makeForm()} datasets={makeDatasets()} onNext={vi.fn()} onBack={vi.fn()} />)
    const nextBtn = screen.getByText('Next: Configure')
    expect(nextBtn.closest('button')!.disabled).toBe(false)
  })

  it('shows helper text when no dataset selected', () => {
    render(<DataStep form={makeForm()} datasets={makeDatasets()} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText(/Select a dataset or switch to paste text/)).toBeTruthy()
  })

  it('hides helper text when data is ready', () => {
    render(<DataStep form={makeForm()} datasets={makeDatasets({ selectedDataset: 'ds-1' })} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.queryByText(/Select a dataset or switch to paste text/)).toBeNull()
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
    expect(screen.getByText('Train from scratch')).toBeTruthy()
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
    render(<ResultsStep checkpoints={makeCheckpoints()} goToTrain={vi.fn()} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText('4. Results')).toBeTruthy()
  })

  it('shows empty state when no checkpoints', () => {
    render(<ResultsStep checkpoints={makeCheckpoints()} goToTrain={vi.fn()} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText(/No checkpoints yet/)).toBeTruthy()
  })

  it('shows checkpoints when available', () => {
    const checkpoints = makeCheckpoints({
      checkpoints: [{ name: 'cp-1', loss: 0.5, tags: ['test'] }],
    })
    render(<ResultsStep checkpoints={checkpoints} goToTrain={vi.fn()} onTest={vi.fn()} addToast={vi.fn()} />)
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
    render(<ResultsStep checkpoints={checkpoints} goToTrain={vi.fn()} onTest={vi.fn()} addToast={vi.fn()} />)
    screen.getByText('Load').closest('button')!.click()
    expect(handleLoadCheckpoint).toHaveBeenCalledWith('cp-1', expect.any(Function))
  })

  it('calls goToTrain when Train more is clicked', () => {
    const goToTrain = vi.fn()
    render(<ResultsStep checkpoints={makeCheckpoints()} goToTrain={goToTrain} onTest={vi.fn()} addToast={vi.fn()} />)
    screen.getByText('Train more').closest('button')!.click()
    expect(goToTrain).toHaveBeenCalled()
  })

  it('calls onTest when Test model is clicked', () => {
    const onTest = vi.fn()
    const checkpoints = makeCheckpoints({ checkpoints: [{ name: 'cp-1', loss: 0.5 }] })
    render(<ResultsStep checkpoints={checkpoints} goToTrain={vi.fn()} onTest={onTest} addToast={vi.fn()} />)
    screen.getByText('Test model').closest('button')!.click()
    expect(onTest).toHaveBeenCalled()
  })

  it('hides Test model button when no checkpoints', () => {
    render(<ResultsStep checkpoints={makeCheckpoints()} goToTrain={vi.fn()} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.queryByText('Test model')).toBeFalsy()
  })

  it('shows checkpoint count', () => {
    const checkpoints = makeCheckpoints({
      checkpoints: [
        { name: 'cp-1', loss: 0.5 },
        { name: 'cp-2', loss: 0.3 },
      ],
    })
    render(<ResultsStep checkpoints={checkpoints} goToTrain={vi.fn()} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText('2 checkpoint(s) saved')).toBeTruthy()
  })
})
