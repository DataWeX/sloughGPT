// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

import { TrainStep } from './TrainStep'
import type { TrainingFormState } from '@/hooks/useTrainingForm'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'

const form: TrainingFormState = {
  method: 'distill',
  inputMode: 'dataset',
  textInput: '',
  showAdvanced: false,
  algo: 'bpe',
  trainingEpochs: 10,
  trainingLR: 0.001,
  trainingBatchSize: 32,
  availableModels: [],
  selectedModel: '',
  useLoRA: false,
  visualVisionEncoder: '',
  visualLLM: '',
  visualStage1Epochs: 0,
  visualStage2Epochs: 0,
  nativeEmbed: 128,
  nativeLayers: 2,
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
  datasets: [],
  selectedDataset: 'shakespeare',
  loadingDatasets: false,
  importModalOpen: false,
  datasetPreview: null,
  setSelectedDataset: vi.fn(),
  setImportModalOpen: vi.fn(),
  setDatasetPreview: vi.fn(),
  fetchDatasets: vi.fn(),
}

describe('TrainStep', () => {
  afterEach(cleanup)

  it('renders the step title', () => {
    render(<TrainStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('3. Train')).toBeDefined()
  })

  it('displays method', () => {
    render(<TrainStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Train from scratch')).toBeDefined()
  })

  it('displays dataset name', () => {
    render(<TrainStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('shakespeare')).toBeDefined()
  })

  it('displays epochs and batch size', () => {
    render(<TrainStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Epochs:')).toBeDefined()
    expect(screen.getByText('Batch size:')).toBeDefined()
  })

  it('shows LoRA enabled when set', () => {
    const loraForm = { ...form, method: 'finetune' as const, useLoRA: true, selectedModel: 'gpt2' }
    render(<TrainStep form={loraForm} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('enabled')).toBeDefined()
  })

  it('has Back button', () => {
    render(<TrainStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Back')).toBeDefined()
  })

  it('displays learning rate', () => {
    render(<TrainStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Learning rate:')).toBeDefined()
  })

  it('hides dataset when not selected', () => {
    const ds = { ...datasets, selectedDataset: '' }
    render(<TrainStep form={form} datasets={ds} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.queryByText(/Dataset:/)).toBeNull()
  })

  it('shows native architecture rows for native method', () => {
    const nativeForm = { ...form, method: 'native' as const, nativeEmbed: 256, nativeLayers: 4, nativeHeads: 8, nativeBlockSize: 128 }
    render(<TrainStep form={nativeForm} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Native SloNet')).toBeDefined()
    expect(screen.getByText('Embed dim:')).toBeDefined()
    expect(screen.getByText('256')).toBeDefined()
    expect(screen.getByText('Layers:')).toBeDefined()
    expect(screen.getByText('Heads:')).toBeDefined()
    expect(screen.getByText('Block size:')).toBeDefined()
  })

  it('shows vision params and hides batch/LR for vlm method', () => {
    const vlmForm = {
      ...form,
      method: 'vlm' as const,
      visualVisionEncoder: 'siglip',
      visualLLM: 'qwen',
      visualStage1Epochs: 1,
      visualStage2Epochs: 2,
    }
    render(<TrainStep form={vlmForm} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Vision (image + text)')).toBeDefined()
    expect(screen.getByText('Vision encoder:')).toBeDefined()
    expect(screen.getByText('Language model:')).toBeDefined()
    expect(screen.getByText('Stage 1 epochs:')).toBeDefined()
    expect(screen.getByText('Stage 2 epochs:')).toBeDefined()
    expect(screen.queryByText('Batch size:')).toBeNull()
    expect(screen.queryByText('Learning rate:')).toBeNull()
  })

  it('shows base model for finetune method', () => {
    const finetuneForm = { ...form, method: 'finetune' as const, selectedModel: 'gpt2' }
    render(<TrainStep form={finetuneForm} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Continue training')).toBeDefined()
    expect(screen.getByText('Base model:')).toBeDefined()
  })

  it('shows pasted-text source and length in text mode', () => {
    const textForm = { ...form, inputMode: 'text' as const, textInput: 'some text' }
    render(<TrainStep form={textForm} datasets={{ ...datasets, selectedDataset: '' }} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Source:')).toBeDefined()
    expect(screen.getByText(/Pasted text \(9 chars\)/)).toBeDefined()
  })
})
