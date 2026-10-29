// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, cleanup, act } from '@testing-library/react'

const mockStartAutoTrain = vi.fn()
vi.mock('@/lib/controllers', () => ({
  trainingJobsController: {
    startAutoTrain: (...args: unknown[]) => mockStartAutoTrain(...args),
  },
}))
vi.mock('@/lib/training-facade', () => ({
  trainingFacade: {
    jobs: {
      startAutoTrain: (...args: unknown[]) => mockStartAutoTrain(...args),
    },
  },
}))

const mockReadTraining = vi.fn()
const mockWriteTraining = vi.fn()
vi.mock('@/lib/app-shell', () => ({
  readTraining: (...args: unknown[]) => mockReadTraining(...args),
  writeTraining: (...args: unknown[]) => mockWriteTraining(...args),
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { child: () => ({ error: vi.fn() }) },
}))

// The hook now subscribes through createSSEStream (transport-managed: base URL,
// bearer token, reconnect backoff all live in fetchSSE), so the seam to drive
// in tests is that stream's callbacks rather than a global EventSource stub.
interface StreamOptions {
  url: string
  onEvent: (envelope: unknown) => void
  onOpen?: () => void
  onClose?: () => void
  onError?: (error: Error) => void
  maxReconnects?: number
}

const mockStreams: StreamOptions[] = []
const mockStart = vi.fn()
const mockStop = vi.fn()

vi.mock('@/lib/sse-client', () => ({
  createSSEStream: (options: StreamOptions) => {
    mockStreams.push(options)
    return { start: mockStart, stop: mockStop, connected: true }
  },
}))

/** Options handed to the most recent createSSEStream call. */
const lastStream = (): StreamOptions => mockStreams[mockStreams.length - 1]

beforeEach(() => {
  vi.clearAllMocks()
  mockStreams.length = 0
  mockReadTraining.mockReturnValue({
    phase: 'idle',
    jobId: null,
    lossHistory: [],
    progress: 0,
    loss: null,
    epoch: 0,
    totalEpochs: 0,
    globalStep: 0,
    totalSteps: 0,
    stepsPerSec: null,
    eta: null,
    elapsedSeconds: null,
    checkpoint: null,
    finalLoss: null,
    error: null,
    method: null,
    message: '',
    evalResult: null,
    startTime: null,
    modelPath: null,
    avgQuality: null,
    dataQuality: null,
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

  it('calls startAutoTrain and creates the stream', async () => {
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

  it("subscribes with a relative path (base URL is the transport's job)", async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())

    await act(async () => {
      result.current.startSSETraining({}, vi.fn())
    })

    expect(lastStream().url).toBe('/training/stream')
    // Three strikes, same policy the EventSource hand-rolled before.
    expect(lastStream().maxReconnects).toBe(3)
    expect(mockStart).toHaveBeenCalled()
  })

  it('parses progress events and writes training state', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())

    await act(async () => {
      result.current.startSSETraining({}, vi.fn())
    })

    await act(() => {
      lastStream().onEvent({
        stream: 'auto-train',
        phase: 'TRAINING',
        data: { progress: 50, global_step: 100, loss: 0.3 },
        meta: { epoch: 2, total_epochs: 10 },
      })
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

    await act(() => {
      lastStream().onEvent({
        stream: 'auto-train',
        status: 'complete',
        data: { checkpoint: 'cp1', final_loss: 0.1 },
      })
    })

    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({ phase: 'complete', checkpoint: 'cp1' }),
    )
    expect(addToast).toHaveBeenCalledWith('Training complete', 'success')
    expect(onCheckpoint).toHaveBeenCalled()
    expect(mockStop).toHaveBeenCalled()
  })

  it('handles error event', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())
    const addToast = vi.fn()

    await act(async () => {
      result.current.startSSETraining({}, addToast)
    })

    await act(() => {
      lastStream().onEvent({ stream: 'auto-train', status: 'error' })
    })

    expect(mockWriteTraining).toHaveBeenCalledWith(expect.objectContaining({ phase: 'error' }))
    expect(addToast).toHaveBeenCalledWith('Training failed', 'error')
  })

  it('ignores events from other streams', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())

    await act(async () => {
      result.current.startSSETraining({}, vi.fn())
    })

    const callsBefore = mockWriteTraining.mock.calls.length
    await act(() => {
      lastStream().onEvent({ stream: 'training', phase: 'progress' })
    })

    expect(mockWriteTraining.mock.calls.length).toBe(callsBefore)
  })

  it('leaves reconnecting streams alone on error', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())

    await act(async () => {
      result.current.startSSETraining({}, vi.fn())
    })

    // The transport owns retry/backoff; an error while it retries must not
    // tear the subscription down from our side.
    await act(() => {
      lastStream().onError?.(new Error('transient'))
    })
    expect(mockStop).not.toHaveBeenCalled()
  })

  it('reports connection loss once reconnects are exhausted', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())
    const addToast = vi.fn()

    await act(async () => {
      result.current.startSSETraining({}, addToast)
    })

    await act(() => {
      lastStream().onClose?.()
    })

    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({ phase: 'error', error: 'Connection lost' }),
    )
    expect(addToast).toHaveBeenCalledWith('Connection lost during training', 'error')
  })

  it('stays quiet when the close was intentional', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingStream())
    const addToast = vi.fn()

    await act(async () => {
      result.current.startSSETraining({}, addToast)
    })

    await act(() => {
      result.current.closeStream()
    })
    await act(() => {
      lastStream().onClose?.()
    })

    expect(mockWriteTraining).not.toHaveBeenCalledWith(
      expect.objectContaining({ error: 'Connection lost' }),
    )
    expect(addToast).not.toHaveBeenCalledWith('Connection lost during training', 'error')
  })

  it('accumulates loss history', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    mockReadTraining.mockReturnValue({
      phase: 'TRAINING',
      jobId: 'j1',
      lossHistory: [{ step: 1, loss: 0.5 }],
      progress: 0,
      loss: 0.5,
      epoch: 0,
      totalEpochs: 0,
      globalStep: 0,
      totalSteps: 0,
      stepsPerSec: null,
      eta: null,
      elapsedSeconds: null,
      checkpoint: null,
      finalLoss: null,
      error: null,
      method: null,
      message: '',
      evalResult: null,
      startTime: null,
      modelPath: null,
      avgQuality: null,
      dataQuality: null,
    })
    const { result } = renderHook(() => useTrainingStream())

    await act(async () => {
      result.current.startSSETraining({}, vi.fn())
    })

    await act(() => {
      lastStream().onEvent({ stream: 'auto-train', data: { loss: 0.3 } })
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

    expect(addToast).toHaveBeenCalledWith('Could not start training: fail', 'error')
  })

  it('stops the stream on unmount', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result, unmount } = renderHook(() => useTrainingStream())

    await act(async () => {
      result.current.startSSETraining({}, vi.fn())
    })

    unmount()
    expect(mockStop).toHaveBeenCalled()
  })
})
