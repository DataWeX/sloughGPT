/**
 */
import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import { useTrainingDatasets } from './useTrainingDatasets'
import { ApiError } from '@/lib/http-client'

const mockList = vi.fn()
const mockCreate = vi.fn()
const mockPreview = vi.fn()
const mockAddData = vi.fn()
const mockFilesList = vi.fn()
const mockGetDetail = vi.fn()
vi.mock('@/lib/controllers', () => ({
  datasetController: {
    list: (...args: unknown[]) => mockList(...args),
    create: (...args: unknown[]) => mockCreate(...args),
    preview: (...args: unknown[]) => mockPreview(...args),
    addData: (...args: unknown[]) => mockAddData(...args),
  },
  filesController: {
    list: (...args: unknown[]) => mockFilesList(...args),
    getDetail: (...args: unknown[]) => mockGetDetail(...args),
  },
}))

const noop = () => {}

const MOCK_DATASETS = [
  {
    id: '1',
    name: 'ds1',
    description: 'first',
    total_samples: 100,
    file_count: 1,
    total_chars: 5000,
    imported_at: '2024-01-01',
  },
  {
    id: '2',
    name: 'ds2',
    description: 'second',
    total_samples: 200,
    file_count: 2,
    total_chars: 10000,
    imported_at: '2024-01-02',
  },
]

const MOCK_FILE = {
  id: '1712345678_train-src_txt',
  filename: 'train-src.txt',
  size: 85,
  content_type: '.txt',
  uploaded_at: '2026-09-27T12:00:00.000Z',
  ingested: true,
}

beforeEach(() => {
  mockFilesList.mockResolvedValue([])
  mockAddData.mockResolvedValue(undefined)
  mockCreate.mockResolvedValue({ id: 'train-src' })
  mockGetDetail.mockResolvedValue({ text: 'fox jumps over dog' })
  mockPreview.mockRejectedValue(new ApiError('not found', 404))
})

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
    await act(async () => {
      await result.current.fetchDatasets()
    })
    expect(result.current.datasets).toEqual(MOCK_DATASETS)
    expect(result.current.loadingDatasets).toBe(false)
  })

  it('fetchDatasets shows toast on error', async () => {
    const addToast = vi.fn()
    mockList.mockRejectedValue(new Error('fail'))
    const { result } = renderHook(() => useTrainingDatasets(addToast))
    await act(async () => {
      await result.current.fetchDatasets()
    })
    expect(result.current.datasets).toEqual([])
    expect(addToast).toHaveBeenCalledWith('Could not fetch datasets: fail', 'error')
  })

  it('fetchDatasets merges uploaded files as promote-on-select entries', async () => {
    mockList.mockResolvedValue(MOCK_DATASETS)
    mockFilesList.mockResolvedValue([MOCK_FILE])
    const { result } = renderHook(() => useTrainingDatasets(noop))
    await act(async () => {
      await result.current.fetchDatasets()
    })
    expect(result.current.datasets).toHaveLength(3)
    const entry = result.current.datasets[0]
    expect(entry).toMatchObject({
      id: MOCK_FILE.id,
      name: 'train-src.txt',
      source: 'my file',
      kind: 'dataset',
      fromFile: true,
    })
  })

  it('fetchDatasets hides a file when a dataset with its sanitized name exists', async () => {
    mockList.mockResolvedValue([{ id: 'train-src', name: 'train-src', source: 'api' }])
    mockFilesList.mockResolvedValue([MOCK_FILE])
    const { result } = renderHook(() => useTrainingDatasets(noop))
    await act(async () => {
      await result.current.fetchDatasets()
    })
    expect(result.current.datasets).toHaveLength(1)
    expect(result.current.datasets[0].id).toBe('train-src')
  })

  it('fetchDatasets tolerates files failure and still loads datasets', async () => {
    mockList.mockResolvedValue(MOCK_DATASETS)
    mockFilesList.mockRejectedValue(new Error('files down'))
    const addToast = vi.fn()
    const { result } = renderHook(() => useTrainingDatasets(addToast))
    await act(async () => {
      await result.current.fetchDatasets()
    })
    expect(result.current.datasets).toEqual(MOCK_DATASETS)
    expect(addToast).not.toHaveBeenCalled()
  })

  it('setSelectedDataset updates selectedDataset', async () => {
    const { result } = renderHook(() => useTrainingDatasets(noop))
    await act(async () => {
      await result.current.setSelectedDataset('abc')
    })
    expect(result.current.selectedDataset).toBe('abc')
  })

  it('selecting an uploaded file promotes it to a dataset', async () => {
    mockList.mockResolvedValue(MOCK_DATASETS)
    mockFilesList.mockResolvedValue([MOCK_FILE])
    const { result } = renderHook(() => useTrainingDatasets(noop))
    await act(async () => {
      await result.current.fetchDatasets()
    })
    await act(async () => {
      await result.current.setSelectedDataset(MOCK_FILE.id)
    })
    expect(mockCreate).toHaveBeenCalledWith({ name: 'train-src' })
    expect(mockAddData).toHaveBeenCalledWith('train-src', ['fox jumps over dog'])
    expect(result.current.selectedDataset).toBe('train-src')
    expect(result.current.datasets.some((d) => d.id === MOCK_FILE.id)).toBe(false)
    expect(result.current.datasets.some((d) => d.id === 'train-src' && !d.fromFile)).toBe(true)
  })

  it('re-select skips addData when the dataset already has samples', async () => {
    mockList.mockResolvedValue(MOCK_DATASETS)
    mockFilesList.mockResolvedValue([MOCK_FILE])
    mockPreview.mockResolvedValue({
      dataset_id: 'train-src',
      samples: [],
      total_samples: 3,
      total_chars: 60,
      languages: {},
    })
    const { result } = renderHook(() => useTrainingDatasets(noop))
    await act(async () => {
      await result.current.fetchDatasets()
    })
    await act(async () => {
      await result.current.setSelectedDataset(MOCK_FILE.id)
    })
    expect(mockCreate).toHaveBeenCalled()
    expect(mockAddData).not.toHaveBeenCalled()
    expect(result.current.selectedDataset).toBe('train-src')
  })

  it('rejects a file with no readable text and leaves selection unchanged', async () => {
    mockList.mockResolvedValue(MOCK_DATASETS)
    mockFilesList.mockResolvedValue([MOCK_FILE])
    mockGetDetail.mockResolvedValue(null)
    const addToast = vi.fn()
    const { result } = renderHook(() => useTrainingDatasets(addToast))
    await act(async () => {
      await result.current.fetchDatasets()
    })
    await act(async () => {
      await result.current.setSelectedDataset(MOCK_FILE.id)
    })
    expect(result.current.selectedDataset).toBe('')
    expect(mockCreate).not.toHaveBeenCalled()
    expect(addToast).toHaveBeenCalledWith(
      'Could not use this file as a dataset: Could not read this file',
      'error',
    )
  })

  it('setImportModalOpen toggles import modal', () => {
    const { result } = renderHook(() => useTrainingDatasets(noop))
    act(() => result.current.setImportModalOpen(true))
    expect(result.current.importModalOpen).toBe(true)
  })

  it('setDatasetPreview stores preview data', () => {
    const { result } = renderHook(() => useTrainingDatasets(noop))
    const preview = {
      dataset_id: 'test',
      samples: [{ content: 'test', path: '', language: 'en', size: 4 }],
      total_samples: 1,
      total_chars: 4,
      languages: { en: 1 },
    }
    act(() => result.current.setDatasetPreview(preview))
    expect(result.current.datasetPreview).toEqual(preview)
  })

  it('loadingDatasets is true during fetch and false after', async () => {
    mockList.mockImplementation(() => new Promise((r) => setTimeout(r, 10)))
    const { result } = renderHook(() => useTrainingDatasets(noop))
    let promise: Promise<void>
    act(() => {
      promise = result.current.fetchDatasets()
    })
    expect(result.current.loadingDatasets).toBe(true)
    await act(async () => {
      await promise
    })
    expect(result.current.loadingDatasets).toBe(false)
  })
})
