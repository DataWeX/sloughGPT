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

  it('visualInference posts to /visual/generate', async () => {
    apiClient.apiPost.mockResolvedValue({ text: 'desc', tokens_generated: 10, elapsed_ms: 100 })
    const result = await visualController.visualInference('base64data', 'What is this?')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/visual/generate', {
      image_base64: 'base64data',
      prompt: 'What is this?',
      max_new_tokens: 256,
      temperature: 0.7,
      top_p: 0.9,
    })
    expect(result.text).toBe('desc')
  })

  it('getVisualStatus gets /visual/status', async () => {
    apiClient.apiGet.mockResolvedValue({ loaded: true, model: 'vlm' })
    const result = await visualController.getVisualStatus()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/visual/status')
    expect(result.loaded).toBe(true)
  })

  it('startVisualTrain posts to /visual/train', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', job_id: 'j1', data_path: '/p', output_dir: '/o' })
    const result = await visualController.startVisualTrain({ data_path: '/data' })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/visual/train', { data_path: '/data' })
    expect(result.job_id).toBe('j1')
  })

  it('getVisualTrainStatus gets /visual/train/status', async () => {
    apiClient.apiGet.mockResolvedValue({ status: 'running', job_id: 'j1', progress: 50, current_stage: 'stage1', total_steps: 100, current_step: 50, current_loss: 1.2, result: null, error: null })
    const result = await visualController.getVisualTrainStatus()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/visual/train/status')
    expect(result.current_loss).toBe(1.2)
  })

  it('loadVisualModel posts to /visual/load', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', message: 'loaded' })
    const result = await visualController.loadVisualModel('models/visual-finetuned')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/visual/load', { model_dir: 'models/visual-finetuned' })
    expect(result.status).toBe('ok')
  })

  it('loadVisualModel uses default model_dir when not provided', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', message: 'loaded' })
    await visualController.loadVisualModel()
    expect(apiClient.apiPost).toHaveBeenCalledWith('/visual/load', { model_dir: 'models/visual-finetuned' })
  })

  it('triggerDPO posts to /visual/dpo', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', message: 'started', job_id: 'j1' })
    const result = await visualController.triggerDPO({ max_pairs: 10 })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/visual/dpo', { max_pairs: 10 })
    expect(result.job_id).toBe('j1')
  })

  it('triggerDPO sends empty object when no data', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', message: 'started', job_id: 'j2' })
    await visualController.triggerDPO()
    expect(apiClient.apiPost).toHaveBeenCalledWith('/visual/dpo', {})
  })

  it('getDPOStatus gets /visual/dpo/status', async () => {
    apiClient.apiGet.mockResolvedValue({ status: 'completed', last_run: '2024-01-01', result: {}, accepted_count: 5, rejected_count: 1 })
    const result = await visualController.getDPOStatus()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/visual/dpo/status')
    expect(result.accepted_count).toBe(5)
  })

  it('listCheckpoints gets /visual/checkpoints', async () => {
    apiClient.apiGet.mockResolvedValue({ checkpoints: [{ name: 'cp1', path: '/p', size_mb: 10, created_at: '2024-01-01', soul_name: 'default', lineage: 'main', llm: 'gpt2', final_loss: 0.5, total_steps: 100, mean_accuracy: 0.9, description: 'test' }] })
    const result = await visualController.listCheckpoints()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/visual/checkpoints')
    expect(result.checkpoints).toHaveLength(1)
  })

  it('deleteCheckpoint deletes /visual/checkpoints/{name}', async () => {
    apiClient.apiDelete.mockResolvedValue({ status: 'deleted', name: 'cp1' })
    const result = await visualController.deleteCheckpoint('cp1')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/visual/checkpoints/cp1')
    expect(result.status).toBe('deleted')
  })

  it('deleteCheckpoint encodes name', async () => {
    apiClient.apiDelete.mockResolvedValue({ status: 'deleted', name: 'my cp' })
    await visualController.deleteCheckpoint('my cp')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/visual/checkpoints/my%20cp')
  })

  it('loadCheckpoint posts to /visual/checkpoints/{name}/load', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', name: 'cp1', path: '/p' })
    const result = await visualController.loadCheckpoint('cp1')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/visual/checkpoints/cp1/load')
    expect(result.path).toBe('/p')
  })
})
