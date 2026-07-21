import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

const {
  mockStartAutoTrain, mockStopAutoTrain, mockStartHFFineTune, mockStartVisualTrain,
  mockStartTurboTrain, mockListJobs,
} = vi.hoisted(() => ({
  mockStartAutoTrain: vi.fn(),
  mockStopAutoTrain: vi.fn(() => Promise.resolve()),
  mockStartHFFineTune: vi.fn(),
  mockStartVisualTrain: vi.fn(),
  mockStartTurboTrain: vi.fn(),
  mockListJobs: vi.fn(),
}))

vi.mock('@/lib/controllers', () => ({
  trainingJobsController: {
    startAutoTrain: mockStartAutoTrain,
    stopAutoTrain: mockStopAutoTrain,
    startHFFineTune: mockStartHFFineTune,
    startVisualTrain: mockStartVisualTrain,
    startTurboTrain: mockStartTurboTrain,
    list: mockListJobs,
    pauseTraining: vi.fn(),
    resumeTraining: vi.fn(),
  },
}))

const mockAddToast = vi.fn()

import { useTrainingSession } from './useTrainingSession'

class MockEventSource {
  onmessage: ((e: MessageEvent) => void) | null = null
  onerror: ((e: Event) => void) | null = null
  readyState = 1
  close = vi.fn()
  static CONNECTING = 0
  static OPEN = 1
  static CLOSED = 2
  constructor(public url: string) {}
  dispatchMessage(data: string) { this.onmessage?.(new MessageEvent('message', { data })) }
  dispatchError() { this.onerror?.(new Event('error')) }
}

describe('useTrainingSession', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    globalThis.EventSource = MockEventSource as any
    mockStartAutoTrain.mockResolvedValue(undefined)
    mockStartHFFineTune.mockRejectedValue(new Error('fail'))
    mockListJobs.mockResolvedValue([])
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
    act(() => { result.current.startSSETraining({ soul: 'friendly' }, mockAddToast) })

    expect(mockStartAutoTrain).toHaveBeenCalledWith({ soul: 'friendly' })
    const es = (globalThis as any).__lastES as MockEventSource | null
    if (es) {
      es.dispatchMessage(JSON.stringify({ stream: 'auto-train', phase: 'TRAIN', data: { loss: 0.5, progress: 50 }, meta: { epoch: 1, total_epochs: 10 } }))
      expect(result.current.phase).toBe('TRAIN')
      expect(result.current.loss).toBe(0.5)
      expect(result.current.progress).toBe(50)
      expect(result.current.epoch).toBe(1)
      expect(result.current.totalEpochs).toBe(10)
    }
  })

  it('startSSETraining handles complete status', async () => {
    mockStartAutoTrain.mockResolvedValue(undefined)
    const onCheckpointUpdate = vi.fn()
    const { result } = renderHook(() => useTrainingSession())
    act(() => { result.current.startSSETraining({ soul: 'friendly' }, mockAddToast, onCheckpointUpdate) })

    const es = (globalThis as any).__lastES as MockEventSource | null
    if (es) {
      es.dispatchMessage(JSON.stringify({ stream: 'auto-train', phase: 'COMPLETE', status: 'complete', data: { checkpoint: 'ckpt1', final_loss: 0.3, epochs: 5 } }))
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
    mockStartHFFineTune.mockResolvedValue({ job_id: 'job-1', message: 'Queued' })
    mockListJobs.mockResolvedValue([
      { id: 'job-1', status: 'completed', result: { model_path: '/model/final', final_loss: 1.2 } },
    ])

    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startFineTune({ model: 'gpt2', dataset: 'data', epochs: 3, batchSize: 4, lr: 0.001, useLoRA: false }, mockAddToast) })
    await act(async () => { vi.advanceTimersByTime(3000) })

    expect(result.current.phase).toBe('complete')
    expect(result.current.finetunedModelPath).toBe('/model/final')
    expect(result.current.finetunedModelLoss).toBe(1.2)
    vi.useRealTimers()
  })

  it('startFineTune handles failure', async () => {
    vi.useFakeTimers()
    mockStartHFFineTune.mockResolvedValue({ job_id: 'job-2', message: 'Queued' })
    mockListJobs.mockResolvedValue([
      { id: 'job-2', status: 'failed', error: 'OOM' },
    ])

    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startFineTune({ model: 'gpt2', dataset: 'data', epochs: 3, batchSize: 4, lr: 0.001, useLoRA: false }, mockAddToast) })
    await act(async () => { vi.advanceTimersByTime(3000) })

    expect(result.current.phase).toBe('error')
    expect(mockAddToast).toHaveBeenCalledWith('OOM', 'error')
    vi.useRealTimers()
  })

  it('startTurboTrain succeeds', async () => {
    mockStartTurboTrain.mockResolvedValue({ status: 'ok', final_loss: 0.5, total_steps: 100 })
    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startTurboTrain('ds-1', { epochs: 5, lr: 1e-3, embed: 128, heads: 4, layers: 2 }, mockAddToast) })
    expect(result.current.turboPhase).toBe('complete')
    expect(result.current.turboResult?.final_loss).toBe(0.5)
    expect(result.current.turboResult?.total_steps).toBe(100)
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
    mockListJobs.mockResolvedValue([
      { id: 'visual-1', status: 'completed', model_path: '/visual/final', loss: 0.8, output_dir: '/out', sou_path: '/out/model.sou' },
    ])

    const { result } = renderHook(() => useTrainingSession())
    await act(async () => { result.current.startVisualTraining({ dataset: 'visual_data', visionEncoder: 'vit', llm: 'gpt2', stage1Epochs: 2, stage2Epochs: 2, useLoRA: true }, mockAddToast) })
    await act(async () => { vi.advanceTimersByTime(3000) })

    expect(result.current.phase).toBe('complete')
    expect(result.current.visualOutputDir).toBe('/out')
    expect(result.current.visualSouPath).toBe('/out/model.sou')
    expect(mockAddToast).toHaveBeenCalledWith('Image model training complete', 'success')
    vi.useRealTimers()
  })

  it('trainingRunning is true during active training', () => {
    const { result } = renderHook(() => useTrainingSession())
    expect(result.current.trainingRunning).toBe(false)
    act(() => { result.current.setPhase('TRAINING') })
    expect(result.current.trainingRunning).toBe(true)
    act(() => { result.current.setPhase('complete') })
    expect(result.current.trainingRunning).toBe(false)
  })
})
