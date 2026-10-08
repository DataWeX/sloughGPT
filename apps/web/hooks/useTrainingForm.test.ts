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
  extractErrorMessage: (e: unknown, fallback: string) =>
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
      expect(addToast).toHaveBeenCalledWith('Select a dataset or paste text to train on', 'error')
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

    it('threads the resume row into the start body (name + exact path)', async () => {
      const checkpoints = {
        jobs: [],
        checkpoints: [
          { name: 'twin.soul', path: 'models/auto-training/twin.soul' },
          { name: 'addressless' },
        ],
        fetchCheckpoints: vi.fn(),
        fetchJobs: vi.fn(),
      } as any
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), makeSession(), checkpoints, addToast),
      )
      await act(async () => {
        await result.current.startTraining('models/auto-training/twin.soul')
      })
      expect(mockTrainingJobsController.startAutoTrain).toHaveBeenCalledWith(
        expect.objectContaining({
          checkpoint_name: 'twin.soul',
          checkpoint_path: 'models/auto-training/twin.soul',
        }),
      )
    })

    it('sends a bare name untouched when no row carries that address (legacy)', async () => {
      const checkpoints = {
        jobs: [],
        checkpoints: [{ name: 'known', path: 'models/known.soul' }],
        fetchCheckpoints: vi.fn(),
        fetchJobs: vi.fn(),
      } as any
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), makeSession(), checkpoints, addToast),
      )
      await act(async () => {
        await result.current.startTraining('orphan-name')
      })
      const body = (mockTrainingJobsController.startAutoTrain as ReturnType<typeof vi.fn>).mock
        .calls[0][0]
      expect(body.checkpoint_name).toBe('orphan-name')
      expect(body).not.toHaveProperty('checkpoint_path')
    })

    it('omits the checkpoint fields entirely for a fresh run', async () => {
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), makeSession(), makeCheckpoints(), addToast),
      )
      await act(async () => {
        await result.current.startTraining()
      })
      const body = (mockTrainingJobsController.startAutoTrain as ReturnType<typeof vi.fn>).mock
        .calls[0][0]
      expect(body).not.toHaveProperty('checkpoint_name')
      expect(body).not.toHaveProperty('checkpoint_path')
    })

    it('sends the CURRENT native architecture to startAutoTrain (no stale closure)', async () => {
      // session/checkpoints hoisted OUT of renderHook: stable identities across
      // renders. The fresh-literal-inside-renderHook fixture recreates the
      // callback every render (new `checkpoints` identity is a dep), hiding
      // exactly the staleness this test exists to pin.
      const session = makeSession()
      const checkpoints = makeCheckpoints()
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), session, checkpoints, addToast),
      )
      // Method first (it IS a dep, so the callback is rebuilt here), then the
      // architecture sliders — exactly the UI order. If the sliders are
      // missing from the deps, this callback keeps the values it captured at
      // setMethod time and Start trains the wrong model size.
      act(() => result.current.setMethod('native'))
      act(() => result.current.setNativeEmbed(256))
      act(() => result.current.setNativeHeads(6))
      await act(async () => {
        await result.current.startTraining()
      })
      expect(mockTrainingJobsController.startAutoTrain).toHaveBeenCalledWith(
        expect.objectContaining({ n_embed: 256, n_head: 6 }),
      )
    })

    it('sends the CURRENT LoRA params to the fine-tune job (no stale closure)', async () => {
      const session = makeSession()
      const checkpoints = makeCheckpoints()
      const { result } = renderHook(() =>
        useTrainingForm(makeDatasets('ds1'), session, checkpoints, addToast),
      )
      act(() => result.current.setMethod('finetune'))
      act(() => result.current.setLoraRank(32))
      act(() => result.current.setLoraAlpha(64))
      await act(async () => {
        await result.current.startTraining()
      })
      expect(session.startFineTune).toHaveBeenCalledWith(
        expect.objectContaining({ loraRank: 32, loraAlpha: 64 }),
        expect.anything(),
        expect.any(Function),
      )
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
})
