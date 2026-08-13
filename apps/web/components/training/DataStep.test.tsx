// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@/components/training/DatasetSelector', () => ({
  DatasetSelector: (props: any) => <div data-testid="dataset-selector">{props.value || 'none'}</div>,
}))

import { DataStep } from './DataStep'
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
  selectedDataset: '',
  loadingDatasets: false,
  importModalOpen: false,
  datasetPreview: null,
  setSelectedDataset: vi.fn(),
  setImportModalOpen: vi.fn(),
  setDatasetPreview: vi.fn(),
  fetchDatasets: vi.fn(),
}

describe('DataStep', () => {
  afterEach(cleanup)

  it('renders the step title', () => {
    render(<DataStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText(/Pick your data/)).toBeDefined()
  })

  it('renders DatasetSelector', () => {
    render(<DataStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByTestId('dataset-selector')).toBeDefined()
  })

  it('shows Next button', () => {
    render(<DataStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText(/Next/)).toBeDefined()
  })

  it('shows hint when no dataset selected', () => {
    render(<DataStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText(/Select a dataset/)).toBeDefined()
  })

  it('shows preview when dataset has samples', () => {
    const datasetsWithPreview: UseTrainingDatasetsReturn = {
      ...datasets,
      datasetPreview: {
        dataset_id: 'ds-1',
        samples: [
          { path: 'a.txt', language: 'text', content: 'line 1', size: 6 },
          { path: 'b.txt', language: 'text', content: 'line 2', size: 6 },
        ],
        total_samples: 100,
        total_chars: 5000,
        languages: { text: 100 },
      },
    }
    render(<DataStep form={form} datasets={datasetsWithPreview} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Preview')).toBeDefined()
    expect(screen.getByText('line 1')).toBeDefined()
  })

  it('hides hint when dataset is selected', () => {
    const ds: UseTrainingDatasetsReturn = { ...datasets, selectedDataset: 'shakespeare' }
    render(<DataStep form={form} datasets={ds} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.queryByText(/Select a dataset/)).toBeNull()
  })

  it('passes selectedDataset to DatasetSelector', () => {
    const ds: UseTrainingDatasetsReturn = { ...datasets, selectedDataset: 'my-dataset' }
    render(<DataStep form={form} datasets={ds} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByTestId('dataset-selector')).toHaveTextContent('my-dataset')
  })
})
