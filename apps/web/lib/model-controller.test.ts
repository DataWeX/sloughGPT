import { beforeEach, describe, expect, it, vi } from 'vitest'

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { modelController, streamModelEvents } from './model-controller'

describe('modelController', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('list returns mapped models', async () => {
    apiClient.apiGet.mockResolvedValue({ models: ['gpt2', { id: 'bert', name: 'BERT', type: 'huggingface', size_mb: 440 }] })
    const result = await modelController.list()
    expect(result).toHaveLength(2)
    expect(result[0]).toEqual({ id: 'gpt2', name: 'gpt2', type: 'huggingface' })
    expect(result[1]).toEqual({ id: 'bert', name: 'BERT', type: 'huggingface', size_mb: 440 })
  })

  it('list returns empty on error', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('fail'))
    const result = await modelController.list()
    expect(result).toEqual([])
  })

  it('load posts to /models/load', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok' })
    const result = await modelController.load('gpt2')
    expect(result.status).toBe('ok')
  })

  it('load throws on error', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'error', error: 'fail' })
    await expect(modelController.load('x')).rejects.toThrow('fail')
  })

  it('status maps from health', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: true, model_type: 'gpt2', device: 'cpu' })
    const result = await modelController.status()
    expect(result).toEqual({ loaded: true, model_type: 'gpt2', device: 'cpu' })
  })

  it('unloadModel posts to /models/unload', async () => {
    apiClient.apiPost.mockResolvedValue({})
    const result = await modelController.unloadModel('gpt2')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/models/unload')
  })

  it('isLoaded checks loaded state', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: true, model_type: 'gpt2' })
    expect(await modelController.isLoaded()).toBe(true)
  })
})
