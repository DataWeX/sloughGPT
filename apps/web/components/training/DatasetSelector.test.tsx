import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { datasetLabel } from './DatasetSelector'
import type { Dataset } from '@/lib/dataset-controller'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/components/DatasetImportModal', () => ({
  DatasetImportModal: () => <div data-testid="import-modal" />,
}))

import { DatasetSelector } from './DatasetSelector'

const makeDataset = (overrides: Partial<Dataset> = {}): Dataset => ({
  id: 'ds-1',
  name: 'shakespeare',
  source: 'local',
  type: 'text',
  size: 2048,
  samples: 100,
  created_at: '2026-01-01',
  ...overrides,
})

const makeProps = (overrides: Partial<UseTrainingDatasetsReturn> = {}): { datasets: UseTrainingDatasetsReturn } & Record<string, any> => ({
  datasets: {
    datasets: [makeDataset()],
    selectedDataset: '',
    setSelectedDataset: vi.fn(),
    importModalOpen: false,
    setImportModalOpen: vi.fn(),
    fetchDatasets: vi.fn().mockResolvedValue(undefined),
    datasetPreview: null,
    setDatasetPreview: vi.fn(),
    loading: false,
    ...overrides,
  } as UseTrainingDatasetsReturn,
})

describe('datasetLabel', () => {
  it('formats basic dataset with size', () => {
    const ds = makeDataset({ size: 2048 })
    expect(datasetLabel(ds)).toBe('shakespeare · 100 samples · local · 2.0 KB')
  })

  it('formats dataset without size', () => {
    const ds = makeDataset({ size: undefined })
    expect(datasetLabel(ds)).toBe('shakespeare · 100 samples · local')
  })

  it('formats dataset with 0 samples', () => {
    const ds = makeDataset({ samples: 0 })
    expect(datasetLabel(ds)).toBe('shakespeare · local · 2.0 KB')
  })

  it('formats VLM dataset', () => {
    const ds = makeDataset({
      type: 'vlm',
      vlm_metadata: { image_count: 50, total_tokens: 1000 },
    } as any)
    expect(datasetLabel(ds)).toBe('shakespeare · VLM · 50 images · local · 2.0 KB')
  })

  it('formats dataset without samples field', () => {
    const ds = makeDataset({ samples: undefined })
    expect(datasetLabel(ds)).toBe('shakespeare · local · 2.0 KB')
  })
})

describe('DatasetSelector', () => {
  afterEach(cleanup)

  it('shows empty state when no datasets', () => {
    const props = makeProps({ datasets: [] as any })
    render(<DatasetSelector {...props} value="" onChange={vi.fn()} />)
    expect(screen.getByText(/No datasets/)).toBeDefined()
  })

  it('shows Import button in empty state', () => {
    const props = makeProps({ datasets: [] as any })
    render(<DatasetSelector {...props} value="" onChange={vi.fn()} />)
    expect(screen.getByText('+ Import')).toBeDefined()
  })

  it('renders select with datasets', () => {
    const props = makeProps()
    render(<DatasetSelector {...props} value="ds-1" onChange={vi.fn()} />)
    expect(screen.getByRole('combobox')).toBeDefined()
  })

  it('shows Import button when showImport is true', () => {
    const props = makeProps()
    render(<DatasetSelector {...props} value="ds-1" onChange={vi.fn()} showImport />)
    const buttons = screen.getAllByText('+ Import')
    expect(buttons.length).toBeGreaterThanOrEqual(1)
  })

  it('hides Import button when showImport is false', () => {
    const props = makeProps()
    render(<DatasetSelector {...props} value="ds-1" onChange={vi.fn()} showImport={false} />)
    expect(screen.queryAllByText('+ Import').length).toBe(0)
  })

  it('renders import modal', () => {
    const props = makeProps()
    render(<DatasetSelector {...props} value="ds-1" onChange={vi.fn()} />)
    expect(screen.getByTestId('import-modal')).toBeDefined()
  })
})
