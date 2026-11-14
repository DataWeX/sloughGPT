// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, type MockInstance } from 'vitest'
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

const { mockModelController, mockTrainingJobsController } = vi.hoisted(() => ({
  mockModelController: {
    list: vi.fn().mockResolvedValue([{ id: 'gpt2' }, { id: 'qwen' }]),
  },
  mockTrainingJobsController: {
    startAutoTrain: vi.fn().mockResolvedValue({ status: 'started', job_id: 'job_test123' }),
  },
}))

vi.mock('@/lib/controllers', () => ({
  modelController: mockModelController,
  trainingJobsController: mockTrainingJobsController,
}))
vi.mock('@/lib/training-facade', () => ({
  trainingFacade: {
    jobs: {
      startAutoTrain: mockTrainingJobsController.startAutoTrain,
    },
  },
}))

vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: (e: unknown, fallback = 'Unknown error') =>
    e instanceof Error ? e.message : fallback,
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
    startSSETraining: vi.fn(),
    startStandardPoll: vi.fn(),
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
        useTrainingForm(makeDatasets('ds1'), makeSession('TRAINING'), makeCheckpoints(), addToast),
      )
      expect(result.current.canStart).toBe(false)
    })

    it('is false when no dataset selected in dataset mode', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(null), makeSession(), makeCheckpoints(), addToast),
      )
      expect(result.current.canStart).toBe(false)
    })

    it('is false when no text in text mode', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(null), makeSession(), makeCheckpoints(), addToast),
      )
      act(() => result.current.setInputMode('text'))
      expect(result.current.canStart).toBe(false)
    })

    it('is false when finetune selected but no model', async () => {
      mockModelController.list.mockResolvedValueOnce([])
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), makeSession(), makeCheckpoints(), addToast),
      )
      act(() => result.current.setMethod('finetune'))
      // selectedModel defaults to '' when no models available
      expect(result.current.canStart).toBe(false)
    })

    it('is false when VLM selected but no dataset', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(null), makeSession(), makeCheckpoints(), addToast),
      )
      act(() => result.current.setMethod('vlm'))
      expect(result.current.canStart).toBe(false)
    })

    it('is true when dataset selected in distill mode', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), makeSession(), makeCheckpoints(), addToast),
      )
      expect(result.current.canStart).toBe(true)
    })

    it('is true when text provided in text mode', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(null), makeSession(), makeCheckpoints(), addToast),
      )
      act(() => result.current.setInputMode('text'))
      act(() => result.current.setTextInput('hello world'))
      expect(result.current.canStart).toBe(true)
    })

    it('is true when VLM with dataset', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), makeSession(), makeCheckpoints(), addToast),
      )
      act(() => result.current.setMethod('vlm'))
      expect(result.current.canStart).toBe(true)
    })
  })

  describe('startTraining', () => {
    it('shows error toast when no data and no checkpoint', async () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(null), makeSession(), makeCheckpoints(), addToast),
      )
      await act(async () => {
        await result.current.startTraining()
      })
      expect(addToast).toHaveBeenCalledWith('Select a dataset, paste text, or list manifests to train on', 'error')
    })

    it('shows error for VLM without dataset (even with text)', async () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(null), makeSession(), makeCheckpoints(), addToast),
      )
      act(() => result.current.setMethod('vlm'))
      act(() => result.current.setInputMode('text'))
      act(() => result.current.setTextInput('some text'))
      await act(async () => {
        await result.current.startTraining()
      })
      expect(addToast).toHaveBeenCalledWith(
        'Vision model training requires a dataset with image-text pairs',
        'error',
      )
    })

    it('starts autotrain for distill method and polls the job', async () => {
      // Distill goes through trainingJobsController.startAutoTrain, then
      // polls the returned job via the session; checkpoints refresh when
      // the poll completes.
      const session = makeSession()
      const checkpoints = makeCheckpoints()
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), session, checkpoints, addToast),
      )
      await act(async () => {
        await result.current.startTraining()
      })
      expect(mockTrainingJobsController.startAutoTrain).toHaveBeenCalled()
      expect(addToast).toHaveBeenCalledWith('Training started', 'info')
      expect(session.startStandardPoll).toHaveBeenCalledWith(
        'job_test123',
        expect.objectContaining({ addToast }),
      )
      const onComplete = (session.startStandardPoll as ReturnType<typeof vi.fn>).mock.calls[0][1]
        ?.onComplete as (() => void) | undefined
      expect(onComplete).toBeDefined()
      await act(async () => {
        onComplete?.()
      })
      expect(checkpoints.fetchCheckpoints).toHaveBeenCalled()
    })

    it('calls startFineTune for finetune method', async () => {
      const session = makeSession()
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), session, makeCheckpoints(), addToast),
      )
      act(() => result.current.setMethod('finetune'))
      await act(async () => {
        await result.current.startTraining()
      })
      expect(session.startFineTune).toHaveBeenCalled()
    })

    it('calls startVisualTraining for VLM method', async () => {
      const session = makeSession()
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), session, makeCheckpoints(), addToast),
      )
      act(() => result.current.setMethod('vlm'))
      await act(async () => {
        await result.current.startTraining()
      })
      expect(session.startVisualTraining).toHaveBeenCalled()
    })

    it('labels native training slonet-native regardless of selected model', async () => {
      // job_cbc6162a: `model: selectedModel || 'slonet-native'` let a selected
      // Qwen model ride along on method=native, so the job record claimed
      // Qwen/Qwen2.5-0.5B-Instruct while a native slonet transformer trained.
      const session = makeSession()
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), session, makeCheckpoints(), addToast),
      )
      act(() => result.current.setMethod('native'))
      act(() => result.current.setSelectedModel('Qwen/Qwen2.5-0.5B-Instruct'))
      await act(async () => {
        await result.current.startTraining()
      })
      expect(mockTrainingJobsController.startAutoTrain).toHaveBeenCalled()
      const body = (mockTrainingJobsController.startAutoTrain as ReturnType<typeof vi.fn>).mock
        .calls[0][0] as Record<string, unknown>
      expect(body.model).toBe('slonet-native')
    })
  })

  describe('defaults', () => {
    it('defaults to distill method', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(), makeSession(), makeCheckpoints(), addToast),
      )
      expect(result.current.method).toBe('distill')
    })

    it('defaults to dataset input mode', () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(), makeSession(), makeCheckpoints(), addToast),
      )
      expect(result.current.inputMode).toBe('dataset')
    })

    it('loads available models on mount', async () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(), makeSession(), makeCheckpoints(), addToast),
      )
      // Wait for model list to load
      await act(async () => {
        await new Promise((r) => setTimeout(r, 10))
      })
      expect(result.current.availableModels).toEqual(['gpt2', 'qwen'])
    })
  })

  describe('ETA pre-flight guard', () => {
    let confirmSpy: MockInstance<typeof window.confirm>

    function makeDatasetsWithSize(selectedDataset: string, size: number) {
      return {
        ...makeDatasets(selectedDataset),
        datasets: [
          {
            id: selectedDataset,
            name: selectedDataset,
            source: 'test',
            size,
            created_at: '2026-01-01T00:00:00Z',
          },
        ],
      }
    }

    beforeEach(() => {
      confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    })

    it('confirms before starting an absurd-steps job', async () => {
      const { result } = renderHook(() =>
        useTrainingForm(
          makeDatasetsWithSize('ds1', 5_000_000),
          makeSession(),
          makeCheckpoints(),
          addToast,
        ),
      )
      act(() => result.current.setMethod('native'))
      act(() => result.current.setTrainingEpochs(500))
      await act(async () => {
        await result.current.startTraining()
      })
      expect(confirmSpy).toHaveBeenCalledTimes(1)
      expect(String(confirmSpy.mock.calls[0][0])).toMatch(/steps/)
      expect(mockTrainingJobsController.startAutoTrain).toHaveBeenCalled()
    })

    it('declining aborts before the optimistic row and the API call', async () => {
      confirmSpy.mockReturnValue(false)
      const { result } = renderHook(() =>
        useTrainingForm(
          makeDatasetsWithSize('ds1', 5_000_000),
          makeSession(),
          makeCheckpoints(),
          addToast,
        ),
      )
      act(() => result.current.setMethod('native'))
      act(() => result.current.setTrainingEpochs(500))
      await act(async () => {
        await result.current.startTraining()
      })
      expect(confirmSpy).toHaveBeenCalledTimes(1)
      expect(result.current.allJobs).toHaveLength(0)
      expect(mockTrainingJobsController.startAutoTrain).not.toHaveBeenCalled()
      expect(addToast).not.toHaveBeenCalledWith('Training started', 'info')
    })

    it('a small job starts without confirmation', async () => {
      const { result } = renderHook(() =>
        useTrainingForm(
          makeDatasetsWithSize('ds1', 50_000),
          makeSession(),
          makeCheckpoints(),
          addToast,
        ),
      )
      await act(async () => {
        await result.current.startTraining()
      })
      expect(confirmSpy).not.toHaveBeenCalled()
      expect(mockTrainingJobsController.startAutoTrain).toHaveBeenCalled()
    })

    it('skips the guard when the dataset size is unknown', async () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), makeSession(), makeCheckpoints(), addToast),
      )
      await act(async () => {
        await result.current.startTraining()
      })
      expect(confirmSpy).not.toHaveBeenCalled()
      expect(mockTrainingJobsController.startAutoTrain).toHaveBeenCalled()
    })

    it('projects text-mode input as well', async () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets(null), makeSession(), makeCheckpoints(), addToast),
      )
      act(() => result.current.setMethod('native'))
      act(() => result.current.setInputMode('text'))
      act(() => result.current.setTextInput('x'.repeat(1_000_000)))
      act(() => result.current.setTrainingBatchSize(1))
      act(() => result.current.setTrainingEpochs(500))
      await act(async () => {
        await result.current.startTraining()
      })
      expect(confirmSpy).toHaveBeenCalledTimes(1)
      expect(mockTrainingJobsController.startAutoTrain).toHaveBeenCalled()
    })
  })

  describe('saved-config sanitization', () => {
    it('clamps out-of-range hyperparameters when loading the saved config', async () => {
      // A legacy/persisted config with out-of-range values (e.g. the
      // reported lr=1-style corruption, or an unbounded epochs entry) must
      // not re-enter form state verbatim.
      chatDBMock.getKV.mockResolvedValueOnce({
        method: 'distill',
        inputMode: 'dataset',
        trainingEpochs: 9999,
        trainingLR: 5,
        trainingBatchSize: 99999,
      })
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), makeSession(), makeCheckpoints(), addToast),
      )
      await act(async () => {
        await new Promise((r) => setTimeout(r, 10))
      })
      expect(result.current.trainingEpochs).toBe(1000)
      expect(result.current.trainingLR).toBe(1)
      expect(result.current.trainingBatchSize).toBe(512)
    })
  })

  describe('startTraining validation limits', () => {
    it('blocks NaN epochs with a limit toast', async () => {
      // NaN slipped past the old `< 1 || > 500` guards (NaN comparisons
      // are always false) and started a garbage job.
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), makeSession(), makeCheckpoints(), addToast),
      )
      act(() => result.current.setTrainingEpochs(Number.NaN))
      await act(async () => {
        await result.current.startTraining()
      })
      expect(addToast).toHaveBeenCalledWith('Epochs must be 1–1000', 'error')
      expect(mockTrainingJobsController.startAutoTrain).not.toHaveBeenCalled()
    })

    it('blocks out-of-range batch size with a limit toast', async () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), makeSession(), makeCheckpoints(), addToast),
      )
      act(() => result.current.setTrainingBatchSize(99999))
      await act(async () => {
        await result.current.startTraining()
      })
      expect(addToast).toHaveBeenCalledWith('Batch size must be 1–512', 'error')
      expect(mockTrainingJobsController.startAutoTrain).not.toHaveBeenCalled()
    })
  })
})
