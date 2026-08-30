// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useTrainingForm } from './useTrainingForm'

const { chatDBMock } = vi.hoisted(() => {
  const chatDBMock = {
    getKV: vi.fn().mockResolvedValue(undefined),
    setKV: vi.fn().mockResolvedValue(undefined),
    deleteKV: vi.fn().mockResolvedValue(undefined),
  }
  return { chatDBMock }
})

vi.mock('@/lib/db', () => ({
  chatDB: chatDBMock,
}))

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

vi.mock('@/lib/controllers', () => ({
  modelController: {
    list: vi.fn().mockResolvedValue([{ id: 'gpt2' }, { id: 'qwen' }]),
  },
  trainingJobsController: {
    startAutoTrain: vi.fn().mockResolvedValue({ status: 'started' }),
  },
}))

vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: (e: unknown, fallback: string) => (e instanceof Error ? e.message : fallback),
}))

function makeDatasets(selectedDataset: string | null = null) {
  return {
    selectedDataset,
    datasets: [],
    loading: false,
    error: null,
    fetchDatasets: vi.fn(),
    selectDataset: vi.fn(),
    deleteDataset: vi.fn(),
    importDataset: vi.fn(),
  } as any
}

function makeSession(phase = 'idle') {
  return {
    phase,
    trainingRunning: phase !== 'idle' && phase !== 'complete' && phase !== 'error',
    startFineTune: vi.fn(),
    startVisualTraining: vi.fn(),
    startTurboTrain: vi.fn(),
  } as any
}

function makeCheckpoints() {
  return {
    jobs: [],
    checkpoints: [],
    fetchCheckpoints: vi.fn(),
    fetchJobs: vi.fn(),
  } as any
}

const addToast = vi.fn()

beforeEach(() => {
  vi.clearAllMocks()
  mockFetch.mockReset()
  chatDBMock.getKV.mockClear()
  chatDBMock.setKV.mockClear()
  chatDBMock.deleteKV.mockClear()
})

describe('useTrainingForm', () => {
  describe('canStart', () => {
    it('is false when training is running', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), makeSession('TRAINING'), makeCheckpoints(), addToast)
      )
      expect(result.current.canStart).toBe(false)
    })

    it('is false when no dataset selected in dataset mode', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(null), makeSession(), makeCheckpoints(), addToast)
      )
      expect(result.current.canStart).toBe(false)
    })

    it('is false when no text in text mode', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(null), makeSession(), makeCheckpoints(), addToast)
      )
      act(() => result.current.setInputMode('text'))
      expect(result.current.canStart).toBe(false)
    })

    it('is false when finetune selected but no model', async () => {
      vi.mocked((await import('@/lib/controllers')).modelController.list).mockResolvedValueOnce([])
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), makeSession(), makeCheckpoints(), addToast)
      )
      act(() => result.current.setMethod('finetune'))
      // selectedModel defaults to '' when no models available
      expect(result.current.canStart).toBe(false)
    })

    it('is false when VLM selected but no dataset', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(null), makeSession(), makeCheckpoints(), addToast)
      )
      act(() => result.current.setMethod('vlm'))
      expect(result.current.canStart).toBe(false)
    })

    it('is true when dataset selected in distill mode', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), makeSession(), makeCheckpoints(), addToast)
      )
      expect(result.current.canStart).toBe(true)
    })

    it('is true when text provided in text mode', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(null), makeSession(), makeCheckpoints(), addToast)
      )
      act(() => result.current.setInputMode('text'))
      act(() => result.current.setTextInput('hello world'))
      expect(result.current.canStart).toBe(true)
    })

    it('is true when VLM with dataset', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), makeSession(), makeCheckpoints(), addToast)
      )
      act(() => result.current.setMethod('vlm'))
      expect(result.current.canStart).toBe(true)
    })

  })

  describe('startTraining', () => {
    it('shows error toast when no data and no checkpoint', async () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(null), makeSession(), makeCheckpoints(), addToast)
      )
      await act(async () => { await result.current.startTraining() })
      expect(addToast).toHaveBeenCalledWith('Select a dataset or paste text to train on', 'error')
    })

    it('shows error for VLM without dataset (even with text)', async () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(null), makeSession(), makeCheckpoints(), addToast)
      )
      act(() => result.current.setMethod('vlm'))
      act(() => result.current.setInputMode('text'))
      act(() => result.current.setTextInput('some text'))
      await act(async () => { await result.current.startTraining() })
      expect(addToast).toHaveBeenCalledWith(
        'Vision model training requires a dataset with image-text pairs', 'error'
      )
    })

    it('calls trainingJobsController.startAutoTrain for distill method', async () => {
      const session = makeSession()
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), session, makeCheckpoints(), addToast)
      )
      await act(async () => { await result.current.startTraining() })
      const { trainingJobsController } = await import('@/lib/controllers')
      expect(trainingJobsController.startAutoTrain).toHaveBeenCalled()
    })

    it('calls startFineTune for finetune method', async () => {
      const session = makeSession()
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), session, makeCheckpoints(), addToast)
      )
      act(() => result.current.setMethod('finetune'))
      await act(async () => { await result.current.startTraining() })
      expect(session.startFineTune).toHaveBeenCalled()
    })

    it('calls startVisualTraining for VLM method', async () => {
      const session = makeSession()
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), session, makeCheckpoints(), addToast)
      )
      act(() => result.current.setMethod('vlm'))
      await act(async () => { await result.current.startTraining() })
      expect(session.startVisualTraining).toHaveBeenCalled()
    })
  })

  describe('defaults', () => {
    it('defaults to distill method', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(), makeSession(), makeCheckpoints(), addToast)
      )
      expect(result.current.method).toBe('distill')
    })

    it('defaults to dataset input mode', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(), makeSession(), makeCheckpoints(), addToast)
      )
      expect(result.current.inputMode).toBe('dataset')
    })

    it('loads available models on mount', async () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(), makeSession(), makeCheckpoints(), addToast)
      )
      // Wait for model list to load
      await act(async () => { await new Promise(r => setTimeout(r, 10)) })
      expect(result.current.availableModels).toEqual(['gpt2', 'qwen'])
    })
  })
})
