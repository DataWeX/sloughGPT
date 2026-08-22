import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

const {
  mockStartAutoTrain, mockStopAutoTrain, mockCreate, mockStartLoraFinetune,
  mockStartVisualTrain, mockStartTurboTrain, mockGetTurboStatus, mockListJobs, mockGetJob,
} = vi.hoisted(() => ({
  mockStartAutoTrain: vi.fn(),
  mockStopAutoTrain: vi.fn(() => Promise.resolve()),
  mockCreate: vi.fn(),
  mockStartLoraFinetune: vi.fn(),
  mockStartVisualTrain: vi.fn(),
  mockStartTurboTrain: vi.fn(),
  mockGetTurboStatus: vi.fn(),
  mockListJobs: vi.fn(),
  mockGetJob: vi.fn(),
}))

vi.mock('@/lib/controllers', () => ({
  trainingJobsController: {
    startAutoTrain: mockStartAutoTrain,
    stopAutoTrain: mockStopAutoTrain,
    create: mockCreate,
    startLoraFinetune: mockStartLoraFinetune,
    startVisualTrain: mockStartVisualTrain,
    startTurboTrain: mockStartTurboTrain,
    getTurboStatus: mockGetTurboStatus,
    list: mockListJobs,
    get: mockGetJob,
    pauseTraining: vi.fn(),
    resumeTraining: vi.fn(),
  },
}))

const mockAddToast = vi.fn()

import { useTrainingSession } from './useTrainingSession'
import { appShellStore } from '@/lib/app-shell'

class MockEventSource {
  onmessage: ((e: MessageEvent) => void) | null = null
  onerror: ((e: Event) => void) | null = null
  readyState = 1
  close = vi.fn()
  static CONNECTING = 0
  static OPEN = 1
  static CLOSED = 2
  constructor(public url: string) {
    ;(globalThis as any).__lastES = this
  }
  dispatchMessage(data: string) { this.onmessage?.(new MessageEvent('message', { data })) }
  dispatchError() { this.onerror?.(new Event('error')) }
}

describe('useTrainingSession', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    globalThis.EventSource = MockEventSource as any
    ;(globalThis as any).__lastES = null
    appShellStore.getState().resetTraining()
    mockStartAutoTrain.mockResolvedValue(undefined)
    mockCreate.mockRejectedValue(new Error('fail'))
    mockListJobs.mockResolvedValue([])
    mockGetTurboStatus.mockResolvedValue({ status: 'idle' })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('returns default state', () => {
    const { result } = renderHook(() => useTrainingSession())
    expect(result.current.phase).toBe('idle')
    expect(result.current.loss).toBeNull()
    expect(result.current.progress).toBe(0)
    expect(result.current.epoch).toBe(0)
    expect(result.current.message).toBe('')
    expect(result.current.trainingRunning).toBe(false)
    expect(result.current.turboRunning).toBe(false)
  })

  it('resetTraining resets all state', () => {
    const { result } = renderHook(() => useTrainingSession())
    act(() => { result.current.setPhase('TRAINING'); result.current.setLoss(1.5); result.current.setProgress(50) })
    act(() => { result.current.resetTraining() })
    expect(result.current.phase).toBe('idle')
    expect(result.current.loss).toBeNull()
    expect(result.current.progress).toBe(0)
  })

  it('stopTraining calls stop and resets', async () => {
    const { result } = renderHook(() => useTrainingSession())
    act(() => { result.current.setPhase('TRAINING') })
    await act(async () => { result.current.stopTraining() })
    expect(mockStopAutoTrain).toHaveBeenCalled()
    expect(result.current.phase).toBe('idle')
  })

  it('startSSETraining creates EventSource and processes SSE events', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startSSETraining({ soul: 'friendly' }, mockAddToast) })

    expect(mockStartAutoTrain).toHaveBeenCalledWith({ soul: 'friendly' })
    const es = (globalThis as any).__lastES as MockEventSource | null
    expect(es).toBeTruthy()
    if (es) {
      await act(async () => { es.dispatchMessage(JSON.stringify({ stream: 'auto-train', phase: 'TRAIN', data: { loss: 0.5, progress: 50 }, meta: { epoch: 1, total_epochs: 10 } })) })
      expect(result.current.phase).toBe('TRAIN')
      expect(result.current.loss).toBe(0.5)
      expect(result.current.progress).toBe(50)
      expect(result.current.epoch).toBe(1)
      expect(result.current.totalEpochs).toBe(10)
    }
  })

  it('startSSETraining captures step/ETA/speed/elapsed fields', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startSSETraining({ soul: 'friendly' }, mockAddToast) })

    const es = (globalThis as any).__lastES as MockEventSource | null
    expect(es).toBeTruthy()
    if (es) {
      await act(async () => { es.dispatchMessage(JSON.stringify({
        stream: 'auto-train', phase: 'TRAIN',
        data: { progress: 40, global_step: 80, total_steps: 200, steps_per_sec: 4.25, eta_s: 28, elapsed_s: 19 },
      })) })
      expect(result.current.progress).toBe(40)
      expect(result.current.globalStep).toBe(80)
      expect(result.current.totalSteps).toBe(200)
      expect(result.current.stepsPerSec).toBe(4.25)
      expect(result.current.eta).toBe(28)
      expect(result.current.elapsedSeconds).toBe(19)
    }
  })

  it('startSSETraining captures avg_quality from SSE', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startSSETraining({ soul: 'friendly' }, mockAddToast) })

    const es = (globalThis as any).__lastES as MockEventSource | null
    expect(es).toBeTruthy()
    if (es) {
      await act(async () => { es.dispatchMessage(JSON.stringify({
        stream: 'auto-train', phase: 'TRAIN',
        data: { progress: 60, avg_quality: 4.2 },
      })) })
      expect(result.current.avgQuality).toBe(4.2)
    }
  })

  it('startSSETraining passes avg_quality on completion', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startSSETraining({ soul: 'friendly' }, mockAddToast) })

    const es = (globalThis as any).__lastES as MockEventSource | null
    expect(es).toBeTruthy()
    if (es) {
      await act(async () => { es.dispatchMessage(JSON.stringify({
        stream: 'auto-train', phase: 'TRAIN',
        data: { progress: 80, avg_quality: 3.9 },
      })) })
      await act(async () => { es.dispatchMessage(JSON.stringify({
        stream: 'auto-train', phase: 'COMPLETE', status: 'complete',
        data: { checkpoint: 'ckpt1', final_loss: 0.3 },
      })) })
      expect(result.current.avgQuality).toBe(3.9)
      expect(result.current.phase).toBe('complete')
    }
  })

  it('startSSETraining handles complete status', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const onCheckpointUpdate = vi.fn()
    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startSSETraining({ soul: 'friendly' }, mockAddToast, onCheckpointUpdate) })

    const es = (globalThis as any).__lastES as MockEventSource | null
    expect(es).toBeTruthy()
    if (es) {
      await act(async () => { es.dispatchMessage(JSON.stringify({ stream: 'auto-train', phase: 'COMPLETE', status: 'complete', data: { checkpoint: 'ckpt1', final_loss: 0.3 }, meta: { total_epochs: 5 } })) })
      expect(result.current.phase).toBe('complete')
      expect(result.current.distillCheckpoint).toBe('ckpt1')
      expect(result.current.distillFinalLoss).toBe(0.3)
      expect(result.current.distillEpochs).toBe(5)
      expect(mockAddToast).toHaveBeenCalledWith('Training complete', 'success')
      expect(onCheckpointUpdate).toHaveBeenCalled()
    }
  })

  it('startFineTune polls for completion', async () => {
    vi.useFakeTimers()
    mockStartLoraFinetune.mockResolvedValue({ job_id: 'job-1', status: 'started' })
    mockGetJob.mockResolvedValue({
      id: 'job-1', status: 'completed', result: { model_path: '/model/final', final_loss: 1.2 },
    })

    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startFineTune({ model: 'gpt2', dataset: 'data', epochs: 3, batchSize: 4, lr: 0.001, useLoRA: false }, mockAddToast) })
    await act(async () => { await vi.advanceTimersByTimeAsync(3000) })

    expect(result.current.phase).toBe('complete')
    expect(result.current.finetunedModelPath).toBe('/model/final')
    expect(result.current.finetunedModelLoss).toBe(1.2)
  })

  it('startFineTune handles failure', async () => {
    vi.useFakeTimers()
    mockStartLoraFinetune.mockResolvedValue({ job_id: 'job-2', status: 'started' })
    mockGetJob.mockResolvedValue({
      id: 'job-2', status: 'failed', error: 'OOM',
    })

    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startFineTune({ model: 'gpt2', dataset: 'data', epochs: 3, batchSize: 4, lr: 0.001, useLoRA: false }, mockAddToast) })
    await act(async () => { await vi.advanceTimersByTimeAsync(3000) })

    expect(result.current.phase).toBe('error')
    expect(mockAddToast).toHaveBeenCalledWith('OOM', 'error')
  })

  it('startTurboTrain polls for completion with live telemetry', async () => {
    vi.useFakeTimers()
    mockStartTurboTrain.mockResolvedValue({ status: 'started', job_id: 't-1', message: 'Queued' })
    mockGetTurboStatus
      .mockResolvedValueOnce({ status: 'idle' }) // reconcile on mount
      .mockResolvedValueOnce({
        status: 'running', job_id: 't-1', progress: 40, global_step: 40, total_steps: 100,
        steps_per_sec: 4.25, eta_s: 14, elapsed_s: 9, loss: 0.5, avg_quality: 4.1,
      })
      .mockResolvedValue({
        status: 'complete', job_id: 't-1',
        result: { status: 'ok', final_loss: 0.32, total_steps: 100, model_path: '/models/turbo/final.soul', avg_quality: 4.3 },
      })

    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startTurboTrain('ds-1', { epochs: 5, lr: 1e-3, embed: 128, heads: 4, layers: 2 }, mockAddToast) })

    expect(result.current.turboPhase).toBe('training')
    expect(mockAddToast).toHaveBeenCalledWith('Turbo training started', 'info')

    await act(async () => { await vi.advanceTimersByTimeAsync(3000) })
    expect(result.current.turboPhase).toBe('training')
    expect(result.current.progress).toBe(40)
    expect(result.current.globalStep).toBe(40)
    expect(result.current.totalSteps).toBe(100)
    expect(result.current.stepsPerSec).toBe(4.25)
    expect(result.current.eta).toBe(14)
    expect(result.current.elapsedSeconds).toBe(9)
    expect(result.current.loss).toBe(0.5)
    expect(result.current.avgQuality).toBe(4.1)

    await act(async () => { await vi.advanceTimersByTimeAsync(3000) })
    expect(result.current.turboPhase).toBe('complete')
    expect(result.current.turboResult?.final_loss).toBe(0.32)
    expect(result.current.turboResult?.total_steps).toBe(100)
    expect(result.current.progress).toBe(100)
    expect(result.current.avgQuality).toBe(4.3)
    expect(mockAddToast).toHaveBeenCalledWith('Turbo training complete!', 'success')
  })

  it('startTurboTrain polls to error', async () => {
    vi.useFakeTimers()
    mockStartTurboTrain.mockResolvedValue({ status: 'started', job_id: 't-2' })
    mockGetTurboStatus
      .mockResolvedValueOnce({ status: 'idle' }) // reconcile on mount
      .mockResolvedValue({ status: 'error', job_id: 't-2', error: 'GPU out of memory' })

    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startTurboTrain('ds-1', { epochs: 5, lr: 1e-3, embed: 128, heads: 4, layers: 2 }, mockAddToast) })
    await act(async () => { await vi.advanceTimersByTimeAsync(3000) })

    expect(result.current.turboPhase).toBe('error')
    expect(result.current.turboError).toBe('GPU out of memory')
    expect(mockAddToast).toHaveBeenCalledWith('GPU out of memory', 'error')
  })

  it('startTurboTrain handles api error', async () => {
    mockStartTurboTrain.mockResolvedValue({ status: 'error', message: 'Training failed' })
    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startTurboTrain('ds-1', { epochs: 5, lr: 1e-3, embed: 128, heads: 4, layers: 2 }, mockAddToast) })
    expect(result.current.turboPhase).toBe('error')
    expect(result.current.turboError).toBe('Training failed')
  })

  it('startTurboTrain handles exception', async () => {
    mockStartTurboTrain.mockRejectedValue(new Error('Network error'))
    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startTurboTrain('ds-1', { epochs: 5, lr: 1e-3, embed: 128, heads: 4, layers: 2 }, mockAddToast) })
    expect(result.current.turboPhase).toBe('error')
  })

  it('startVisualTraining polls for completion', async () => {
    vi.useFakeTimers()
    mockStartVisualTrain.mockResolvedValue({ job_id: 'visual-1', message: 'Queued' })
    mockGetJob.mockResolvedValue({
      id: 'visual-1', status: 'completed', model_path: '/visual/final', loss: 0.8, output_dir: '/out', sou_path: '/out/model.sou',
    })

    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startVisualTraining({ dataset: 'visual_data', visionEncoder: 'vit', llm: 'gpt2', stage1Epochs: 2, stage2Epochs: 2, useLoRA: true }, mockAddToast) })
    await act(async () => { await vi.advanceTimersByTimeAsync(3000) })

    expect(result.current.phase).toBe('complete')
    expect(result.current.visualOutputDir).toBe('/out')
    expect(result.current.visualSouPath).toBe('/out/model.sou')
    expect(mockAddToast).toHaveBeenCalledWith('Image model training complete', 'success')
  })

  it('trainingRunning is true during active training', () => {
    const { result } = renderHook(() => useTrainingSession())
    expect(result.current.trainingRunning).toBe(false)
    act(() => { result.current.setPhase('TRAINING') })
    expect(result.current.trainingRunning).toBe(true)
    act(() => { result.current.setPhase('complete') })
    expect(result.current.trainingRunning).toBe(false)
  })

  it('reconciles with standard training job on mount', async () => {
    vi.useFakeTimers()
    mockGetTurboStatus.mockResolvedValue({ status: 'idle' })
    mockListJobs.mockResolvedValue([
      { id: 'std-1', status: 'running', method: 'slnet', progress: 45, loss: 1.2, current_epoch: 2, epochs: 10, global_step: 450, total_steps: 1000, steps_per_sec: 3.5, eta_s: 160, elapsed_s: 120 },
    ])
    mockGetJob.mockResolvedValue({ id: 'std-1', status: 'running', progress: 50, loss: 1.1 })

    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })

    expect(result.current.phase).toBe('TRAINING')
    expect(result.current.method).toBe('slnet')
    expect(result.current.progress).toBe(45)
    expect(result.current.loss).toBe(1.2)
    expect(result.current.jobId).toBe('std-1')

    await act(async () => { await vi.advanceTimersByTimeAsync(3000) })
    expect(result.current.progress).toBe(50)
    expect(result.current.loss).toBe(1.1)
  })
})
