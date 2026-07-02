import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mockBlob = new Blob(['fake'], { type: 'image/png' })
const mockFetchResponse = { blob: () => Promise.resolve(mockBlob) }
vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(mockFetchResponse)))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { multimodalController } from './multimodal-controller'

beforeEach(() => { vi.clearAllMocks() })
afterEach(() => { vi.unstubAllGlobals() })

describe('multimodalController', () => {
  it('getCapabilities GETs /multimodal/capabilities', async () => {
    apiClient.apiGet.mockResolvedValue({ speech_to_text: true, image_caption: false, status: 'ready' })
    const cap = await multimodalController.getCapabilities()
    expect(cap.status).toBe('ready')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/capabilities')
  })

  it('getLearningProgress GETs /multimodal/learning-progress', async () => {
    apiClient.apiGet.mockResolvedValue({ images_learned: 5, trained: true, vocab_size: 100, replay_buffer_size: 50 })
    const p = await multimodalController.getLearningProgress()
    expect(p.images_learned).toBe(5)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/learning-progress')
  })

  it('getTrainingReport GETs /multimodal/training-report', async () => {
    apiClient.apiGet.mockResolvedValue({ images_learned: 10, vocab_size: 200, trained: true, caption_history: ['a'], unique_captions: 1, diversity_ratio: 1, accuracy_history: [0.5], mean_accuracy: 0.5, last_accuracy: 0.5, replay_buffer_size: 100 })
    const r = await multimodalController.getTrainingReport()
    expect(r.images_learned).toBe(10)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/training-report')
  })

  it('getTrainingStatus GETs /multimodal/training-status', async () => {
    apiClient.apiGet.mockResolvedValue({ running: false, job_id: null, total: 0, completed: 0, errors: 0, progress_pct: 0, current_caption: '', current_image: '', started_at: null, finished_at: null })
    const s = await multimodalController.getTrainingStatus()
    expect(s.running).toBe(false)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/multimodal/training-status')
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
