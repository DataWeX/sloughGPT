// @vitest-environment jsdom
import { describe, it, expect, vi, beforeAll, beforeEach, afterAll, afterEach } from 'vitest'
import { renderHook, cleanup } from '@testing-library/react'

const mockStart = vi.fn()
const mockStop = vi.fn()
let _onEvent: ((envelope: Record<string, unknown>) => void) | null = null
let _onClose: (() => void) | null = null
let _onError: (() => void) | null = null

vi.mock('@/lib/sse-client', () => ({
  createSSEStream: (opts: Record<string, unknown>) => {
    _onEvent = opts.onEvent as (e: Record<string, unknown>) => void
    _onClose = opts.onClose as () => void
    _onError = opts.onError as () => void
    return { start: mockStart, stop: mockStop, get connected() { return true } }
  },
}))

const mockReadTraining = vi.fn()
const mockWriteTraining = vi.fn()
vi.mock('@/lib/app-shell', () => ({
  readTraining: (...args: unknown[]) => mockReadTraining(...args),
  writeTraining: (...args: unknown[]) => mockWriteTraining(...args),
}))

const mockGetJob = vi.fn()
vi.mock('@/lib/controllers', () => ({
  trainingJobsController: { get: (...args: unknown[]) => mockGetJob(...args) },
}))

import { useLiveTraining, initTrainingStream } from './useLiveTraining'

const S = {
  phase: 'idle', jobId: null, progress: 0, loss: null,
  lossHistory: [] as Array<{ step: number; loss: number }>,
  epoch: 0, totalEpochs: 0, globalStep: 0, totalSteps: 0,
  stepsPerSec: null, eta: null, elapsedSeconds: null,
  checkpoint: null, finalLoss: null, error: null, method: null,
  message: '', evalResult: null, startTime: null, modelPath: null,
  avgQuality: null, dataQuality: null,
}

describe('useLiveTraining hook', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
    _onEvent = null
    _onClose = null
    _onError = null
    mockReadTraining.mockReturnValue({ ...S })
    mockGetJob.mockResolvedValue(null)
  })
  afterEach(() => { cleanup(); vi.useRealTimers() })

  it('returns idle state initially', () => {
    const { result } = renderHook(() => useLiveTraining())
    expect(result.current.isTraining).toBe(false)
  })

  it('starts SSE stream on mount', () => {
    renderHook(() => useLiveTraining())
    expect(mockStart).toHaveBeenCalled()
  })

  it('stops SSE stream on unmount', () => {
    const { unmount } = renderHook(() => useLiveTraining())
    unmount()
    expect(mockStop).toHaveBeenCalled()
  })

  it('reports isTraining from readTraining', () => {
    mockReadTraining.mockReturnValue({ ...S, phase: 'TRAINING' })
    const { result } = renderHook(() => useLiveTraining())
    expect(result.current.isTraining).toBe(true)
  })

  it('reports idle when phase is not TRAINING', () => {
    mockReadTraining.mockReturnValue({ ...S, phase: 'complete' })
    const { result } = renderHook(() => useLiveTraining())
    expect(result.current.isTraining).toBe(false)
  })
})

describe('initTrainingStream SSE events', () => {
  beforeAll(() => {
    vi.useFakeTimers()
    mockReadTraining.mockReturnValue({ ...S })
    mockGetJob.mockResolvedValue(null)
    initTrainingStream()
  })

  beforeEach(() => {
    mockReadTraining.mockReturnValue({ ...S })
    mockGetJob.mockResolvedValue(null)
  })

  afterAll(() => { vi.useRealTimers() })

  it('handles init event with active job', () => {
    _onEvent!({
      stream: 'training', phase: 'init',
      data: { jobs: { job1: { status: 'running', progress: 50, loss: 0.5 } } },
    })
    vi.advanceTimersByTime(16)
    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({ jobId: 'job1', phase: 'TRAINING' }),
    )
  })

  it('handles progress event', () => {
    mockReadTraining.mockReturnValue({ ...S, phase: 'TRAINING', jobId: 'j1' })
    _onEvent!({
      stream: 'training', phase: 'progress',
      data: { job_id: 'j1', progress: 75, train_loss: 0.3, global_step: 100 },
    })
    vi.advanceTimersByTime(16)
    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({ progress: 75, loss: 0.3, globalStep: 100 }),
    )
  })

  it('handles started event', () => {
    _onEvent!({ stream: 'training', phase: 'started', data: { job_id: 'j2' } })
    vi.advanceTimersByTime(16)
    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({ jobId: 'j2', phase: 'TRAINING', progress: 0 }),
    )
  })

  it('handles completed event', () => {
    _onEvent!({
      stream: 'training', phase: 'completed',
      data: { job_id: 'j3', checkpoint: 'cp1', loss: 0.1 },
    })
    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({ phase: 'complete', progress: 100, checkpoint: 'cp1' }),
    )
  })

  it('handles failed event', () => {
    _onEvent!({
      stream: 'training', phase: 'failed',
      data: { job_id: 'j4', error: 'OOM' },
    })
    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({ phase: 'error', error: 'OOM' }),
    )
  })

  it('ignores events from other streams', () => {
    const callsBefore = mockWriteTraining.mock.calls.length
    _onEvent!({ stream: 'operations', phase: 'init', data: {} })
    expect(mockWriteTraining.mock.calls.length).toBe(callsBefore)
  })

  it('starts fallback poll on SSE close when training active', () => {
    mockReadTraining.mockReturnValue({ ...S, phase: 'TRAINING', jobId: 'j1' })
    mockGetJob.mockResolvedValue({ status: 'running', progress: 60 })
    _onClose!()
    vi.advanceTimersByTime(5000)
    expect(mockGetJob).toHaveBeenCalledWith('j1')
  })

  it('starts fallback poll on SSE error when training active', () => {
    mockReadTraining.mockReturnValue({ ...S, phase: 'TRAINING', jobId: 'j1' })
    mockGetJob.mockResolvedValue({ status: 'running', progress: 60 })
    _onError!()
    vi.advanceTimersByTime(5000)
    expect(mockGetJob).toHaveBeenCalledWith('j1')
  })

  it('batchWriteTraining flushes terminal events immediately', () => {
    _onEvent!({
      stream: 'training', phase: 'completed',
      data: { job_id: 'j1', checkpoint: 'cp1' },
    })
    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({ phase: 'complete' }),
    )
  })

  it('fallback poll stops when job completes', async () => {
    mockReadTraining.mockReturnValue({ ...S, phase: 'TRAINING', jobId: 'j1' })
    mockGetJob.mockResolvedValue({ status: 'completed', checkpoint: 'cp1' })
    _onClose!()
    await vi.advanceTimersByTimeAsync(5000)
    expect(mockWriteTraining).toHaveBeenCalledWith(
      expect.objectContaining({ phase: 'complete' }),
    )
  })
})
