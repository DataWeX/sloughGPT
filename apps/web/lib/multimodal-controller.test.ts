import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mockBlob = new Blob(['fake'], { type: 'image/png' })
const mockFetchResponse = { blob: () => Promise.resolve(mockBlob) }
vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(mockFetchResponse)))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { multimodalController } from './multimodal-controller'

const unifiedResponse = {
  engine: { speech_to_text: true, image_caption: true, speech_model: 'whisper', vision_model: 'soulnet', status: 'trained' },
  learning: { images_learned: 10, trained: true, vocab_size: 256, replay_buffer_size: 200, learning_method: 'contrastive + self-training', caption_history: ['a cat', 'a dog'], unique_captions: 2, diversity_ratio: 1.0, accuracy_history: [0.5, 0.9], mean_accuracy: 0.7, last_accuracy: 0.9 },
  batch: { running: false, job_id: null, total: 0, completed: 0, errors: 0, progress_pct: 0, current_caption: '', current_image: '', started_at: null, finished_at: null },
  dpo: { status: 'idle', last_run: null, result: null, accepted_count: 0, rejected_count: 0 },
  video: { status: 'idle', job_id: null, current_epoch: 0, current_step: 0, total_steps: 0, current_loss: null, result: null, error: null },
}

beforeEach(() => { vi.clearAllMocks() })
afterEach(() => { vi.unstubAllGlobals() })

describe('multimodalController', () => {
  it('getCapabilities calls GET /multimodal/status', async () => {
    apiClient.apiGet.mockResolvedValue(unifiedResponse)
    const cap = await multimodalController.getCapabilities()
    expect(cap.status).toBe('trained')
    expect(cap.images_learned).toBe(10)
    expect(cap.learning_method).toBe('contrastive + self-training')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/status')
  })

  it('getLearningProgress calls GET /multimodal/status', async () => {
    apiClient.apiGet.mockResolvedValue(unifiedResponse)
    const p = await multimodalController.getLearningProgress()
    expect(p.images_learned).toBe(10)
    expect(p.vocab_size).toBe(256)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/status')
  })

  it('getTrainingReport calls GET /multimodal/status', async () => {
    apiClient.apiGet.mockResolvedValue(unifiedResponse)
    const r = await multimodalController.getTrainingReport()
    expect(r.caption_history).toHaveLength(2)
    expect(r.unique_captions).toBe(2)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/status')
  })

  it('getTrainingStatus calls GET /multimodal/status', async () => {
    apiClient.apiGet.mockResolvedValue(unifiedResponse)
    const s = await multimodalController.getTrainingStatus()
    expect(s.running).toBe(false)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/status')
  })

  it('getTrainingStatus returns idle state', async () => {
    apiClient.apiGet.mockResolvedValue(unifiedResponse)
    const s = await multimodalController.getTrainingStatus()
    expect(s.running).toBe(false)
    expect(s.progress_pct).toBe(0)
  })

  it('trainImage POSTs dataUrl via fetch then FormData', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', caption: 'cat', confidence: 0.9, images_learned: 1, accuracy: 0.85, supervised: false })
    const result = await multimodalController.trainImage('data:image/png;base64,abc', 'cat.png', 'cat')
    expect(result.status).toBe('ok')
    expect(result.caption).toBe('cat')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/multimodal/train', expect.any(FormData), { raw: true })
  })

  it('trainBatch POSTs files as FormData', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', job_id: 'j1', total_images: 2 })
    const files = [new File(['a'], 'a.png'), new File(['b'], 'b.png')]
    const result = await multimodalController.trainBatch(files)
    expect(result.job_id).toBe('j1')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/multimodal/train-batch', expect.any(FormData), { raw: true })
  })

  it('trainBatchFromDir POSTs dataset_path as FormData', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', job_id: 'j2', total_images: 5 })
    const result = await multimodalController.trainBatchFromDir('/data/images')
    expect(result.total_images).toBe(5)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/multimodal/train-batch', expect.any(FormData), { raw: true })
  })

  it('transcribeAudio POSTs file as FormData', async () => {
    apiClient.apiPost.mockResolvedValue({ text: 'hello world' })
    const result = await multimodalController.transcribeAudio(new File(['audio'], 'test.wav'))
    expect(result.text).toBe('hello world')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/multimodal/transcribe', expect.any(FormData), { raw: true })
  })

  it('generateImage POSTs prompt/steps/guidance_scale as FormData', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', image: 'base64...', prompt: 'cat' })
    const result = await multimodalController.generateImage('cat', 30, 8.0)
    expect(result.status).toBe('ok')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/multimodal/generate-image', expect.any(FormData), { raw: true })
  })

  it('processVideo POSTs file + num_frames as FormData', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', caption: 'walking', num_frames: 16 })
    const result = await multimodalController.processVideo(new File(['vid'], 'test.mp4'), 32)
    expect(result.caption).toBe('walking')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/multimodal/process-video', expect.any(FormData), { raw: true })
  })

  it('synthesizeSpeech POSTs text as FormData', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', audio: 'base64...', text: 'hello', duration_sec: 1.5 })
    const result = await multimodalController.synthesizeSpeech('hello')
    expect(result.duration_sec).toBe(1.5)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/multimodal/synthesize-speech', expect.any(FormData), { raw: true })
  })

  it('analyzeImage POSTs file + optional prompt as FormData', async () => {
    apiClient.apiPost.mockResolvedValue({ caption: 'dog', confidence: 0.95, tags: ['dog'], accuracy: 0.9, supervised: true, images_learned: 1, trained: true, replay_buffer_size: 100, mean_accuracy: 0.9 })
    const result = await multimodalController.analyzeImage(new File(['img'], 'test.png'), 'what is this?')
    expect(result.caption).toBe('dog')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/multimodal/analyze', expect.any(FormData), { raw: true })
  })

  it('resetModel POSTs to /multimodal/reset', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', message: 'reset done' })
    const result = await multimodalController.resetModel()
    expect(result.message).toBe('reset done')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/multimodal/reset')
  })
})
