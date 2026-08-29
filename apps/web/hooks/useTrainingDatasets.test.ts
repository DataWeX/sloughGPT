/**
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import { useTrainingDatasets } from './useTrainingDatasets'

const mockList = vi.fn()
vi.mock('@/lib/controllers', () => ({
  datasetController: { list: (...args: unknown[]) => mockList(...args) },
}))

const noop = () => {}

const MOCK_DATASETS = [
  { id: '1', name: 'ds1', description: 'first', total_samples: 100, file_count: 1, total_chars: 5000, imported_at: '2024-01-01' },
  { id: '2', name: 'ds2', description: 'second', total_samples: 200, file_count: 2, total_chars: 10000, imported_at: '2024-01-02' },
]

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('useTrainingDatasets', () => {
  it('returns default state', () => {
    const { result } = renderHook(() => useTrainingDatasets(noop))
    expect(result.current.datasets).toEqual([])
    expect(result.current.selectedDataset).toBe('')
    expect(result.current.loadingDatasets).toBe(false)
    expect(result.current.importModalOpen).toBe(false)
    expect(result.current.datasetPreview).toBeNull()
  })

  it('fetchDatasets loads datasets from controller', async () => {
    mockList.mockResolvedValue(MOCK_DATASETS)
    const { result } = renderHook(() => useTrainingDatasets(noop))
    await act(async () => { await result.current.fetchDatasets() })
    expect(result.current.datasets).toEqual(MOCK_DATASETS)
    expect(result.current.loadingDatasets).toBe(false)
  })

  it('fetchDatasets shows toast on error', async () => {
    const addToast = vi.fn()
    mockList.mockRejectedValue(new Error('fail'))
    const { result } = renderHook(() => useTrainingDatasets(addToast))
    await act(async () => { await result.current.fetchDatasets() })
    expect(result.current.datasets).toEqual([])
    expect(addToast).toHaveBeenCalledWith('Could not fetch datasets', 'error')
  })

  it('setSelectedDataset updates selectedDataset', () => {
    const { result } = renderHook(() => useTrainingDatasets(noop))
    act(() => result.current.setSelectedDataset('abc'))
    expect(result.current.selectedDataset).toBe('abc')
  })

  it('setImportModalOpen toggles import modal', () => {
    const { result } = renderHook(() => useTrainingDatasets(noop))
    act(() => result.current.setImportModalOpen(true))
    expect(result.current.importModalOpen).toBe(true)
  })

  it('setDatasetPreview stores preview data', () => {
    const { result } = renderHook(() => useTrainingDatasets(noop))
    const preview = { dataset_id: 'test', samples: [{ content: 'test', path: '', language: 'en', size: 4 }], total_samples: 1, total_chars: 4, languages: { en: 1 } }
    act(() => result.current.setDatasetPreview(preview))
    expect(result.current.datasetPreview).toEqual(preview)
  })

  it('loadingDatasets is true during fetch and false after', async () => {
    mockList.mockImplementation(() => new Promise(r => setTimeout(r, 10)))
    const { result } = renderHook(() => useTrainingDatasets(noop))
    let promise: Promise<void>
    act(() => { promise = result.current.fetchDatasets() })
    expect(result.current.loadingDatasets).toBe(true)
    await act(async () => { await promise })
    expect(result.current.loadingDatasets).toBe(false)
  })
})
