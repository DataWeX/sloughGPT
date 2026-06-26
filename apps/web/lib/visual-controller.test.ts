import { beforeEach, describe, expect, it, vi } from 'vitest'

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { visualController } from './visual-controller'

describe('visualController', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('createVisualDataset posts to /multimodal/visual-dataset', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', dataset: 'test', path: '/p', entries: 10, auto_captioned: true })
    const result = await visualController.createVisualDataset('test', '/images')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/multimodal/visual-dataset', {
      name: 'test',
      image_dir: '/images',
      caption_prompt: 'Describe this image in detail.',
      auto_caption: true,
    })
    expect(result.status).toBe('ok')
  })

  it('getVisualStatus gets /visual/status', async () => {
    apiClient.apiGet.mockResolvedValue({ video_training: {}, dpo: {} })
    const result = await visualController.getVisualStatus()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/visual/status')
    expect(result.dpo).toBeDefined()
  })

  it('triggerDPO posts to /visual/dpo', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', steps: 10, avg_loss: null, ppl_before: null, ppl_after: null, ppl_delta_pct: null, pairs_trained: 5, elapsed_seconds: 1.2 })
    const result = await visualController.triggerDPO({ max_pairs: 10 })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/visual/dpo', { max_pairs: 10 })
    expect(result.pairs_trained).toBe(5)
  })

  it('triggerDPO sends empty object when no data', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', steps: 0, avg_loss: null, ppl_before: null, ppl_after: null, ppl_delta_pct: null, pairs_trained: 0, elapsed_seconds: 0.1 })
    await visualController.triggerDPO()
    expect(apiClient.apiPost).toHaveBeenCalledWith('/visual/dpo', {})
  })

  it('getDPOStatus gets /visual/dpo/status', async () => {
    apiClient.apiGet.mockResolvedValue({ status: 'completed', last_run: '2024-01-01', result: {}, accepted_count: 5, rejected_count: 1 })
    const result = await visualController.getDPOStatus()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/visual/dpo/status')
    expect(result.accepted_count).toBe(5)
  })

  it('startVideoTrain posts to /visual/train-video', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'started', job_id: 'j1', data_path: '/p', output_dir: '/o' })
    const result = await visualController.startVideoTrain({ data_path: '/data', epochs: 5 })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/visual/train-video', { data_path: '/data', epochs: 5 })
    expect(result.job_id).toBe('j1')
  })

  it('getVideoTrainStatus gets /visual/train-video/status', async () => {
    apiClient.apiGet.mockResolvedValue({ status: 'running', job_id: 'j1', current_epoch: 1, current_step: 10, total_steps: 100, current_loss: 1.2, result: null, error: null })
    const result = await visualController.getVideoTrainStatus()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/visual/train-video/status')
    expect(result.current_loss).toBe(1.2)
  })

  it('videoInference posts to /visual/video-infer', async () => {
    apiClient.apiPost.mockResolvedValue({ text: 'caption', checkpoint: 'cp1', elapsed_ms: 500 })
    const result = await visualController.videoInference({ video_path: '/v.mp4', max_len: 50 })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/visual/video-infer', { video_path: '/v.mp4', max_len: 50 })
    expect(result.text).toBe('caption')
  })
})
