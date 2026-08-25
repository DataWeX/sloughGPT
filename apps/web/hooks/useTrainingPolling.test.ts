// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, cleanup } from '@testing-library/react'

const mockGetJob = vi.fn()
const mockGetTurboStatus = vi.fn()
vi.mock('@/lib/controllers', () => ({
  trainingJobsController: {
    get: (...args: unknown[]) => mockGetJob(...args),
    getTurboStatus: (...args: unknown[]) => mockGetTurboStatus(...args),
  },
}))

const mockReadTraining = vi.fn()
const mockWriteTraining = vi.fn()
vi.mock('@/lib/app-shell', () => ({
  readTraining: (...args: unknown[]) => mockReadTraining(...args),
  writeTraining: (...args: unknown[]) => mockWriteTraining(...args),
}))

import { useTrainingPolling } from './useTrainingPolling'

beforeEach(() => {
  vi.clearAllMocks()
  vi.useFakeTimers()
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
  vi.useRealTimers()
})

describe('useTrainingPolling', () => {
  it('returns startStandardPoll, startTurboPoll, clearAllPolls', () => {
    const { result } = renderHook(() => useTrainingPolling())
    expect(typeof result.current.startStandardPoll).toBe('function')
    expect(typeof result.current.startTurboPoll).toBe('function')
    expect(typeof result.current.clearAllPolls).toBe('function')
  })

  describe('startStandardPoll', () => {
    it('starts polling and fetches job', async () => {
      mockGetJob.mockResolvedValue({ status: 'running', progress: 50 })
      const { result } = renderHook(() => useTrainingPolling())

      result.current.startStandardPoll('job1')
      await vi.advanceTimersByTimeAsync(3000)

      expect(mockGetJob).toHaveBeenCalledWith('job1')
    })

    it('updates training state on running job', async () => {
      mockGetJob.mockResolvedValue({
        status: 'running', progress: 50, loss: 0.3,
        current_epoch: 2, epochs: 10, global_step: 100, total_steps: 1000,
        steps_per_sec: 5.0, eta_s: 180, elapsed_s: 20, avg_quality: 0.8,
      })
      const { result } = renderHook(() => useTrainingPolling())

      result.current.startStandardPoll('job1')
      await vi.advanceTimersByTimeAsync(3000)

      expect(mockWriteTraining).toHaveBeenCalledWith(
        expect.objectContaining({ progress: 50, loss: 0.3, epoch: 2 }),
      )
    })

    it('completes on completed job', async () => {
      mockGetJob.mockResolvedValue({
        status: 'completed', checkpoint: 'cp1', loss: 0.1,
        avg_quality: 0.9,
      })
      const addToast = vi.fn()
      const onComplete = vi.fn()
      const { result } = renderHook(() => useTrainingPolling())

      result.current.startStandardPoll('job1', { addToast, onComplete, completeMessage: 'Done!' })
      await vi.advanceTimersByTimeAsync(3000)

      expect(mockWriteTraining).toHaveBeenCalledWith(
        expect.objectContaining({ phase: 'complete', checkpoint: 'cp1' }),
      )
      expect(addToast).toHaveBeenCalledWith('Done!', 'success')
      expect(onComplete).toHaveBeenCalled()
    })

    it('handles failed job', async () => {
      mockGetJob.mockResolvedValue({ status: 'failed', error: 'OOM' })
      const addToast = vi.fn()
      const { result } = renderHook(() => useTrainingPolling())

      result.current.startStandardPoll('job1', { addToast })
      await vi.advanceTimersByTimeAsync(3000)

      expect(mockWriteTraining).toHaveBeenCalledWith(
        expect.objectContaining({ phase: 'error', error: 'OOM' }),
      )
      expect(addToast).toHaveBeenCalledWith('OOM', 'error')
    })

    it('stops polling when job not found', async () => {
      mockGetJob.mockResolvedValue(null)
      const { result } = renderHook(() => useTrainingPolling())

      result.current.startStandardPoll('job1')
      await vi.advanceTimersByTimeAsync(3000)

      expect(mockWriteTraining).not.toHaveBeenCalled()
    })

    it('retries on transient errors', async () => {
      mockGetJob
        .mockRejectedValueOnce(new Error('network'))
        .mockResolvedValue({ status: 'running', progress: 60 })
      const { result } = renderHook(() => useTrainingPolling())

      result.current.startStandardPoll('job1')
      await vi.advanceTimersByTimeAsync(3000)

      expect(mockGetJob).toHaveBeenCalledTimes(1)
      expect(mockWriteTraining).not.toHaveBeenCalled()

      await vi.advanceTimersByTimeAsync(30000)

      expect(mockGetJob.mock.calls.length).toBeGreaterThanOrEqual(2)
      expect(mockWriteTraining).toHaveBeenCalledWith(
        expect.objectContaining({ progress: 60 }),
      )
    })

    it('stops after MAX_POLL_RETRIES consecutive errors', async () => {
      mockGetJob.mockRejectedValue(new Error('persistent'))
      const addToast = vi.fn()
      const { result } = renderHook(() => useTrainingPolling())

      result.current.startStandardPoll('job1', { addToast })
      // Exponential backoff: 3s, 6s, 12s, 24s, 30s cap...
      for (let i = 0; i < 11; i++) {
        await vi.advanceTimersByTimeAsync(60000)
      }

      expect(addToast).toHaveBeenCalledWith(
        expect.stringContaining('Lost connection'),
        'error',
      )
    })

    it('accumulates loss history', async () => {
      mockReadTraining.mockReturnValue({
        phase: 'TRAINING', jobId: 'j1',
        lossHistory: [{ step: 1, loss: 0.5 }],
        progress: 0, loss: 0.5, epoch: 0, totalEpochs: 0,
        globalStep: 0, totalSteps: 0, stepsPerSec: null, eta: null,
        elapsedSeconds: null, checkpoint: null, finalLoss: null,
        error: null, method: null, message: '', evalResult: null,
        startTime: null, modelPath: null, avgQuality: null, dataQuality: null,
      })
      mockGetJob.mockResolvedValue({
        status: 'running', loss: 0.3, global_step: 2,
      })
      const { result } = renderHook(() => useTrainingPolling())

      result.current.startStandardPoll('job1')
      await vi.advanceTimersByTimeAsync(3000)

      expect(mockWriteTraining).toHaveBeenCalledWith(
        expect.objectContaining({
          lossHistory: expect.arrayContaining([
            expect.objectContaining({ loss: 0.5 }),
            expect.objectContaining({ loss: 0.3 }),
          ]),
        }),
      )
    })
  })

  describe('startTurboPoll', () => {
    it('starts polling and fetches turbo status', async () => {
      mockGetTurboStatus.mockResolvedValue({ status: 'running', progress: 30 })
      const { result } = renderHook(() => useTrainingPolling())

      result.current.startTurboPoll()
      await vi.advanceTimersByTimeAsync(3000)

      expect(mockGetTurboStatus).toHaveBeenCalled()
    })

    it('completes on turbo complete', async () => {
      mockGetTurboStatus.mockResolvedValue({
        status: 'complete',
        result: { checkpoint: 'tcp1', final_loss: 0.05, model_path: '/models/turbo/final.soul' },
      })
      const addToast = vi.fn()
      const { result } = renderHook(() => useTrainingPolling())

      result.current.startTurboPoll(addToast)
      await vi.advanceTimersByTimeAsync(3000)

      expect(mockWriteTraining).toHaveBeenCalledWith(
        expect.objectContaining({ phase: 'complete', checkpoint: 'tcp1', modelPath: '/models/turbo/final.soul' }),
      )
      expect(addToast).toHaveBeenCalledWith('Turbo training complete!', 'success')
    })

    it('handles turbo error', async () => {
      mockGetTurboStatus.mockResolvedValue({ status: 'error', error: 'GPU OOM' })
      const addToast = vi.fn()
      const { result } = renderHook(() => useTrainingPolling())

      result.current.startTurboPoll(addToast)
      await vi.advanceTimersByTimeAsync(3000)

      expect(mockWriteTraining).toHaveBeenCalledWith(
        expect.objectContaining({ phase: 'error', error: 'GPU OOM' }),
      )
    })

    it('stops and resets phase when turbo goes idle', async () => {
      mockGetTurboStatus.mockResolvedValue({ status: 'idle' })
      const { result } = renderHook(() => useTrainingPolling())

      result.current.startTurboPoll()
      await vi.advanceTimersByTimeAsync(3000)

      expect(mockWriteTraining).toHaveBeenCalledWith(
        expect.objectContaining({ phase: 'idle' }),
      )
    })

    it('stops after MAX_POLL_RETRIES', async () => {
      mockGetTurboStatus.mockRejectedValue(new Error('fail'))
      const addToast = vi.fn()
      const { result } = renderHook(() => useTrainingPolling())

      result.current.startTurboPoll(addToast)
      for (let i = 0; i < 11; i++) {
        await vi.advanceTimersByTimeAsync(3000)
      }

      expect(addToast).toHaveBeenCalledWith(
        expect.stringContaining('Lost connection'),
        'error',
      )
    })
  })

  describe('clearAllPolls', () => {
    it('clears all active polls', async () => {
      mockGetJob.mockResolvedValue({ status: 'running', progress: 50 })
      const { result } = renderHook(() => useTrainingPolling())

      result.current.startStandardPoll('job1')
      await vi.advanceTimersByTimeAsync(3000)
      result.current.clearAllPolls()

      const callsBefore = mockGetJob.mock.calls.length
      await vi.advanceTimersByTimeAsync(6000)
      expect(mockGetJob.mock.calls.length).toBe(callsBefore)
    })
  })

  it('clears polls on unmount', async () => {
    mockGetJob.mockResolvedValue({ status: 'running', progress: 50 })
    const { result, unmount } = renderHook(() => useTrainingPolling())

    result.current.startStandardPoll('job1')
    await vi.advanceTimersByTimeAsync(3000)
    unmount()

    const callsBefore = mockGetJob.mock.calls.length
    await vi.advanceTimersByTimeAsync(6000)
    expect(mockGetJob.mock.calls.length).toBe(callsBefore)
  })
})
