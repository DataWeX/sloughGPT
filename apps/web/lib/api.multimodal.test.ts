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
import { visualController } from './visual-controller'

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

describe('visualController.createVisualDataset', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs /multimodal/visual-dataset with config', async () => {
    apiClient.apiPost.mockResolvedValue({
      status: 'created',
      dataset: 'my-dataset',
      path: '/repo/datasets/my-dataset/corpus.jsonl',
      entries: 15,
      auto_captioned: true,
    })

    const result = await visualController.createVisualDataset('my-dataset', '/data/images')
    expect(result.status).toBe('created')
    expect(result.entries).toBe(15)
    expect(result.auto_captioned).toBe(true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/multimodal/visual-dataset', {
      name: 'my-dataset',
      image_dir: '/data/images',
      caption_prompt: 'Describe this image in detail.',
      auto_caption: true,
    })
  })

  it('allows custom caption prompt', async () => {
    apiClient.apiPost.mockResolvedValue({
      status: 'created',
      dataset: 'custom-dataset',
      path: '/repo/datasets/custom-dataset/corpus.jsonl',
      entries: 5,
      auto_captioned: false,
    })

    const result = await visualController.createVisualDataset('custom-dataset', '/path', 'What do you see?', false)
    expect(result.entries).toBe(5)
    expect(result.auto_captioned).toBe(false)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/multimodal/visual-dataset', {
      name: 'custom-dataset',
      image_dir: '/path',
      caption_prompt: 'What do you see?',
      auto_caption: false,
    })
  })
})

describe('modelController.loadVisualModel', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs /models/visual-load with query params', async () => {
    const { modelController } = await import('./model-controller')
    apiClient.apiPost.mockResolvedValue({
      status: 'loaded',
      model_id: 'visual',
      type: 'visual',
      vision_encoder: 'google/siglip-base-patch16-224',
      llm: 'Qwen/Qwen2.5-0.5B-Instruct',
    })

    const result = await modelController.loadVisualModel('/path/to/visual-dir', 'my-visual')
    expect(result.status).toBe('loaded')
    expect(result.type).toBe('visual')
    expect(result.vision_encoder).toContain('siglip')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/models/visual-load?model_dir=%2Fpath%2Fto%2Fvisual-dir&model_id=my-visual')
  })
})
