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

const unifiedResponse = {
  engine: { speech_to_text: true, image_caption: true, speech_model: 'whisper', vision_model: 'soulnet', status: 'trained' },
  learning: { images_learned: 12, trained: true, vocab_size: 256, replay_buffer_size: 200, learning_method: 'contrastive + self-training', caption_history: ['a cat', 'a dog'], unique_captions: 2, diversity_ratio: 1.0, accuracy_history: [0.5, 0.9], mean_accuracy: 0.7, last_accuracy: 0.9 },
  batch: { running: true, job_id: 'batch_20260101_120000', total: 20, completed: 10, errors: 1, progress_pct: 50, current_caption: 'a landscape', current_image: 'img_001.jpg', started_at: '2026-01-01T12:00:00', finished_at: null },
  dpo: { status: 'idle', last_run: null, result: null, accepted_count: 0, rejected_count: 0 },
  video: { status: 'idle', job_id: null, current_epoch: 0, current_step: 0, total_steps: 0, current_loss: null, result: null, error: null },
}

describe('multimodalController.getCapabilities', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /multimodal/status and maps fields', async () => {
    apiClient.apiGet.mockResolvedValue(unifiedResponse)
    const caps = await multimodalController.getCapabilities()
    expect(caps.speech_to_text).toBe(true)
    expect(caps.images_learned).toBe(12)
    expect(caps.learning_method).toBe('contrastive + self-training')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/status')
  })

  it('maps engine fields', async () => {
    apiClient.apiGet.mockResolvedValue(unifiedResponse)
    const caps = await multimodalController.getCapabilities()
    expect(caps.image_caption).toBe(true)
    expect(caps.speech_model).toBe('whisper')
    expect(caps.vision_model).toBe('soulnet')
  })
})

describe('multimodalController.getLearningProgress', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /multimodal/status and maps learning fields', async () => {
    apiClient.apiGet.mockResolvedValue(unifiedResponse)
    const prog = await multimodalController.getLearningProgress()
    expect(prog.images_learned).toBe(12)
    expect(prog.vocab_size).toBe(256)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/status')
  })

  it('maps accuracy fields', async () => {
    apiClient.apiGet.mockResolvedValue(unifiedResponse)
    const prog = await multimodalController.getLearningProgress()
    expect(prog.trained).toBe(true)
    expect(prog.replay_buffer_size).toBe(200)
  })
})

describe('multimodalController.getTrainingReport', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /multimodal/status and maps report fields', async () => {
    apiClient.apiGet.mockResolvedValue(unifiedResponse)
    const report = await multimodalController.getTrainingReport()
    expect(report.caption_history).toHaveLength(2)
    expect(report.unique_captions).toBe(2)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/status')
  })

  it('maps diversity ratio', async () => {
    apiClient.apiGet.mockResolvedValue(unifiedResponse)
    const report = await multimodalController.getTrainingReport()
    expect(report.diversity_ratio).toBe(1.0)
  })
})

describe('multimodalController.getTrainingStatus', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /multimodal/status and returns batch', async () => {
    apiClient.apiGet.mockResolvedValue(unifiedResponse)
    const s = await multimodalController.getTrainingStatus()
    expect(s.running).toBe(true)
    expect(s.progress_pct).toBe(50)
    expect(s.current_image).toBe('img_001.jpg')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/status')
  })

  it('returns idle state when not running', async () => {
    apiClient.apiGet.mockResolvedValue({
      ...unifiedResponse,
      batch: { running: false, total: 0, completed: 0, errors: 0, progress_pct: 0, current_caption: '', current_image: '' },
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

  it('returns error status on failure', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'error', message: 'no images found' })
    const result = await multimodalController.trainBatchFromDir('/empty')
    expect(result.status).toBe('error')
  })
})
