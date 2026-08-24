// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, cleanup, act } from '@testing-library/react'

const mockStartAutoTrain = vi.fn()
vi.mock('@/lib/controllers', () => ({
  trainingJobsController: {
    startAutoTrain: (...args: unknown[]) => mockStartAutoTrain(...args),
  },
}))

const mockReadTraining = vi.fn()
const mockWriteTraining = vi.fn()
vi.mock('@/lib/app-shell', () => ({
  readTraining: (...args: unknown[]) => mockReadTraining(...args),
  writeTraining: (...args: unknown[]) => mockWriteTraining(...args),
}))

vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { child: () => ({ error: vi.fn() }) },
}))

// Mock EventSource globally
class MockEventSource {
  static CLOSED = 2
  url: string
  onmessage: ((e: MessageEvent) => void) | null = null
  onerror: (() => void) | null = null
  readyState = 1
  close = vi.fn()
  constructor(url: string) {
    this.url = url
    ;(globalThis as Record<string, unknown>).__lastES = this
  }
}
;(globalThis as Record<string, unknown>).EventSource = MockEventSource

beforeEach(() => {
  vi.clearAllMocks()
  mockReadTraining.mockReturnValue({
    phase: 'idle', jobId: null, lossHistory: [],
    progress: 0, loss: null, epoch: 0, totalEpochs: 0,
    globalStep: 0, totalSteps: 0, stepsPerSec: null, eta: null,
    elapsedSeconds: null, checkpoint: null, finalLoss: null,
    error: null, method: null, message: '', evalResult: null,
    startTime: null, modelPath: null, avgQuality: null, dataQuality: null,
  })
})

afterEach(() => {
  cleanup()
})

import { useTrainingStream } from './useTrainingStream'

describe('useTrainingStream', () => {
  it('returns startSSETraining and closeStream', () => {
    const { result } = renderHook(() => useTrainingStream())
    expect(typeof result.current.startSSETraining).toBe('function')
    expect(typeof result.current.closeStream).toBe('function')
  })

  it('calls startAutoTrain and creates EventSource', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())
    const addToast = vi.fn()

    await act(async () => {
      result.current.startSSETraining({ soul: 'test' }, addToast)
    })

    expect(mockStartAutoTrain).toHaveBeenCalledWith({ soul: 'test' })
    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({ phase: 'TRAINING', method: 'slonet' }),
    )
  })

  it('sets up EventSource with correct URL', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())

    await act(async () => {
      result.current.startSSETraining({}, vi.fn())
    })

    const es = (globalThis as Record<string, unknown>).__lastES as MockEventSource
    expect(es.url).toBe('http://localhost:8000/auto-train/stream')
  })

  it('parses progress events and writes training state', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())

    await act(async () => {
      result.current.startSSETraining({}, vi.fn())
    })

    const es = (globalThis as Record<string, unknown>).__lastES as MockEventSource
    await act(() => {
      es.onmessage?.({
        data: JSON.stringify({
          stream: 'auto-train',
          phase: 'training',
          data: { progress: 50, global_step: 100, loss: 0.3 },
          meta: { epoch: 2, total_epochs: 10 },
        }),
      } as MessageEvent)
    })

    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({ progress: 50, globalStep: 100, loss: 0.3 }),
    )
  })

  it('handles completion event', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())
    const addToast = vi.fn()
    const onCheckpoint = vi.fn()

    await act(async () => {
      result.current.startSSETraining({}, addToast, onCheckpoint)
    })

    const es = (globalThis as Record<string, unknown>).__lastES as MockEventSource
    await act(() => {
      es.onmessage?.({
        data: JSON.stringify({
          stream: 'auto-train',
          status: 'complete',
          data: { checkpoint: 'cp1', final_loss: 0.1 },
        }),
      } as MessageEvent)
    })

    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({ phase: 'complete', checkpoint: 'cp1' }),
    )
    expect(addToast).toHaveBeenCalledWith('Training complete', 'success')
    expect(onCheckpoint).toHaveBeenCalled()
    expect(es.close).toHaveBeenCalled()
  })

  it('handles error event', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())
    const addToast = vi.fn()

    await act(async () => {
      result.current.startSSETraining({}, addToast)
    })

    const es = (globalThis as Record<string, unknown>).__lastES as MockEventSource
    await act(() => {
      es.onmessage?.({
        data: JSON.stringify({
          stream: 'auto-train',
          status: 'error',
        }),
      } as MessageEvent)
    })

    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({ phase: 'error' }),
    )
    expect(addToast).toHaveBeenCalledWith('Training failed', 'error')
  })

  it('ignores events from other streams', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())

    await act(async () => {
      result.current.startSSETraining({}, vi.fn())
    })

    const es = (globalThis as Record<string, unknown>).__lastES as MockEventSource
    const callsBefore = mockWriteTraining.mock.calls.length
    await act(() => {
      es.onmessage?.({
        data: JSON.stringify({ stream: 'training', phase: 'progress' }),
      } as MessageEvent)
    })

    expect(mockWriteTraining.mock.calls.length).toBe(callsBefore)
  })

  it('handles SSE errors with retry', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())

    await act(async () => {
      result.current.startSSETraining({}, vi.fn())
    })

    const es = (globalThis as Record<string, unknown>).__lastES as MockEventSource
    es.readyState = 1 // CONNECTING
    await act(() => {
      es.onerror?.()
    })
    // Should not close on first error if readyState is not CLOSED
    expect(es.close).not.toHaveBeenCalled()
  })

  it('handles SSE close after max retries', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())
    const addToast = vi.fn()

    await act(async () => {
      result.current.startSSETraining({}, addToast)
    })

    const es = (globalThis as Record<string, unknown>).__lastES as MockEventSource
    es.readyState = MockEventSource.CLOSED
    await act(() => {
      es.onerror?.()
    })

    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({ phase: 'error', error: 'Connection lost' }),
    )
  })

  it('accumulates loss history', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    mockReadTraining.mockReturnValue({
      phase: 'TRAINING', jobId: 'j1',
      lossHistory: [{ step: 1, loss: 0.5 }],
      progress: 0, loss: 0.5, epoch: 0, totalEpochs: 0,
      globalStep: 0, totalSteps: 0, stepsPerSec: null, eta: null,
      elapsedSeconds: null, checkpoint: null, finalLoss: null,
      error: null, method: null, message: '', evalResult: null,
      startTime: null, modelPath: null, avgQuality: null, dataQuality: null,
    })
    const { result } = renderHook(() => useTrainingStream())

    await act(async () => {
      result.current.startSSETraining({}, vi.fn())
    })

    const es = (globalThis as Record<string, unknown>).__lastES as MockEventSource
    await act(() => {
      es.onmessage?.({
        data: JSON.stringify({
          stream: 'auto-train',
          data: { loss: 0.3 },
        }),
      } as MessageEvent)
    })

    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({
        lossHistory: expect.arrayContaining([
          expect.objectContaining({ loss: 0.5 }),
          expect.objectContaining({ loss: 0.3 }),
        ]),
      }),
    )
  })

  it('handles startAutoTrain failure', async () => {
    mockStartAutoTrain.mockRejectedValue(new Error('fail'))
    const { result } = renderHook(() => useTrainingStream())
    const addToast = vi.fn()

    await act(async () => {
      result.current.startSSETraining({}, addToast)
    })

    expect(addToast).toHaveBeenCalledWith('Could not start training', 'error')
  })

  it('cleans up EventSource on unmount', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result, unmount } = renderHook(() => useTrainingStream())

    await act(async () => {
      result.current.startSSETraining({}, vi.fn())
    })

    const es = (globalThis as Record<string, unknown>).__lastES as MockEventSource
    unmount()
    expect(es.close).toHaveBeenCalled()
  })
})
