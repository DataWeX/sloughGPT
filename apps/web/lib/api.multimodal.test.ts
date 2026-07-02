import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { multimodalController } from './multimodal-controller'

describe('multimodalController.getCapabilities', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /multimodal/capabilities', async () => {
    apiClient.apiGet.mockResolvedValue({
      speech_to_text: true,
      image_caption: true,
      speech_model: 'whisper',
      vision_model: 'soulnet',
      images_learned: 12,
      trained: true,
      replay_buffer_size: 100,
      learning_method: 'contrastive + self-training',
      background_job_running: false,
      status: 'trained',
    })

    const caps = await multimodalController.getCapabilities()
    expect(caps.speech_to_text).toBe(true)
    expect(caps.images_learned).toBe(12)
    expect(caps.learning_method).toBe('contrastive + self-training')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/capabilities')
  })
})

describe('multimodalController.getLearningProgress', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /multimodal/learning-progress', async () => {
    apiClient.apiGet.mockResolvedValue({
      images_learned: 5,
      trained: false,
      vocab_size: 128,
      replay_buffer_size: 50,
    })

    const prog = await multimodalController.getLearningProgress()
    expect(prog.images_learned).toBe(5)
    expect(prog.vocab_size).toBe(128)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/learning-progress')
  })
})

describe('multimodalController.getTrainingReport', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /multimodal/training-report with caption history', async () => {
    apiClient.apiGet.mockResolvedValue({
      images_learned: 10,
      vocab_size: 256,
      replay_buffer_size: 200,
      caption_history: ['a cat', 'a dog', 'a bird'],
      unique_captions: 3,
      diversity_ratio: 1.0,
      trained: true,
    })

    const report = await multimodalController.getTrainingReport()
    expect(report.caption_history).toHaveLength(3)
    expect(report.unique_captions).toBe(3)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/training-report')
  })
})

describe('multimodalController.getTrainingStatus', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /multimodal/training-status', async () => {
    apiClient.apiGet.mockResolvedValue({
      running: true,
      job_id: 'batch_20260101_120000',
      total: 20,
      completed: 10,
      errors: 1,
      progress_pct: 50,
      current_caption: 'a landscape',
      current_image: 'img_001.jpg',
      started_at: '2026-01-01T12:00:00',
      finished_at: null,
    })

    const s = await multimodalController.getTrainingStatus()
    expect(s.running).toBe(true)
    expect(s.progress_pct).toBe(50)
    expect(s.current_image).toBe('img_001.jpg')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/training-status')
  })

  it('returns idle state when not running', async () => {
    apiClient.apiGet.mockResolvedValue({
      running: false,
      total: 0,
      completed: 0,
      errors: 0,
      progress_pct: 0,
      current_caption: '',
      current_image: '',
    })

    const s = await multimodalController.getTrainingStatus()
    expect(s.running).toBe(false)
    expect(s.progress_pct).toBe(0)
  })
})

describe('multimodalController.trainBatchFromDir', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs /multimodal/train-batch with dataset_path', async () => {
    apiClient.apiPost.mockResolvedValue({
      status: 'started',
      job_id: 'batch_20260101_120000',
      total_images: 42,
    })

    const result = await multimodalController.trainBatchFromDir('/path/to/images')
    expect(result.status).toBe('started')
    expect(result.total_images).toBe(42)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/multimodal/train-batch', expect.any(FormData), { raw: true })
  })
})
