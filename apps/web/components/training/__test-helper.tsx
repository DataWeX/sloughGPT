/**
 * Shared test factories for training components.
 *
 * Provides reusable mock builders for TrainingFormState, UseTrainingDatasetsReturn,
 * UseTrainingCheckpointsReturn, and a Dataset factory. All factories accept
 * partial overrides via the spread pattern.
 *
 * Usage:
 *   import { makeForm, makeDatasets, makeCheckpoints, ds } from './__test-helper'
 *   const form = makeForm({ trainingEpochs: 10 })
 */
import { vi } from 'vitest'
import type { Dataset } from '@/lib/dataset-controller'

// ---------------------------------------------------------------------------
// TrainingFormState factory
// ---------------------------------------------------------------------------

export function makeForm(overrides: Record<string, any> = {}) {
  return {
    method: 'distill' as const,
    inputMode: 'dataset' as const,
    textInput: '',
    trainingEpochs: 3,
    trainingLR: 0.001,
    trainingBatchSize: 8,
    availableModels: [] as any[],
    selectedModel: '',
    useLoRA: false,
    loraRank: 8,
    loraAlpha: 16,
    visualVisionEncoder: '',
    visualLLM: '',
    visualStage1Epochs: 1,
    visualStage2Epochs: 2,
    nativeEmbed: 64,
    nativeLayers: 4,
    nativeHeads: 4,
    nativeBlockSize: 128,
    showAdvanced: false,
    algo: 'bpe',
    loadingFinetunedModel: false,
    allJobs: [] as any[],
    customPresets: [] as any[],
    canStart: true,
    setMethod: vi.fn(),
    setInputMode: vi.fn(),
    setTextInput: vi.fn(),
    setTrainingEpochs: vi.fn(),
    setTrainingLR: vi.fn(),
    setTrainingBatchSize: vi.fn(),
    setSelectedModel: vi.fn(),
    setUseLoRA: vi.fn(),
    setLoraRank: vi.fn(),
    setLoraAlpha: vi.fn(),
    setVlmVisionEncoder: vi.fn(),
    setVlmLLM: vi.fn(),
    setVlmStage1Epochs: vi.fn(),
    setVlmStage2Epochs: vi.fn(),
    setNativeEmbed: vi.fn(),
    setNativeLayers: vi.fn(),
    setNativeHeads: vi.fn(),
    setNativeBlockSize: vi.fn(),
    setShowAdvanced: vi.fn(),
    setAlgo: vi.fn(),
    setLoadingFinetunedModel: vi.fn(),
    applyPreset: vi.fn(),
    saveCustomPreset: vi.fn(),
    deleteCustomPreset: vi.fn(),
    startTraining: vi.fn(),
    resumeCheckpoint: '',
    setResumeCheckpoint: vi.fn(),
    ...overrides,
  }
}

// ---------------------------------------------------------------------------
// UseTrainingDatasetsReturn factory
// ---------------------------------------------------------------------------

export function makeDatasets(overrides: Record<string, any> = {}) {
  return {
    datasets: [] as any[],
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

// ---------------------------------------------------------------------------
// UseTrainingCheckpointsReturn factory
// ---------------------------------------------------------------------------

export function makeCheckpoints(overrides: Record<string, any> = {}) {
  return {
    checkpoints: [] as any[],
    loadingCheckpoints: false,
    loadingJobs: false,
    activeCheckpoint: null as any,
    builds: [] as any[],
    loadingBuilds: false,
    jobs: [] as any[],
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

// ---------------------------------------------------------------------------
// Dataset object factory
// ---------------------------------------------------------------------------

export const ds = (overrides: Partial<Dataset> = {}): Dataset => ({
  id: '1',
  name: 'shakespeare',
  type: 'text',
  source: '',
  size: 0,
  created_at: '2026-01-01T00:00:00Z',
  ...overrides,
})

// ---------------------------------------------------------------------------
// Shared strui mock presets
// ---------------------------------------------------------------------------

/** Passthrough component — renders children in a plain div. */
export const passthrough = ({ children }: any) => <div>{children}</div>

/** Button mock — renders a <button> with variant/size data attributes. */
export const mockButton = ({ children, onClick, disabled, variant, size, className }: any) => (
  <button onClick={onClick} disabled={disabled} data-variant={variant} data-size={size} className={className}>{children}</button>
)

/** CardTitle mock — renders children with a data-testid. */
export const mockCardTitle = ({ children }: any) => <div data-testid="card-title">{children}</div>

/** Common strui mock map — pass as `vi.mock('@sloughgpt/strui', () => STRUI_MOCK)` partial merge. */
export const STRUI_PASSTHROUGH_MOCK = {
  Card: passthrough,
  CardContent: passthrough,
  CardHeader: passthrough,
  CardTitle: mockCardTitle,
  Button: mockButton,
}
