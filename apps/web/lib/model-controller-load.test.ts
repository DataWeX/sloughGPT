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

import { modelController } from './model-controller'

describe('modelController.load', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs /models/load with model_id and default device', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'loaded', model: 'gpt2', model_type: 'gpt2' })

    await modelController.load('gpt2')

    expect(apiClient.apiPost).toHaveBeenCalledWith(
      '/models/load',
      { model_id: 'gpt2', device: 'auto' },
    )
  })

  it('throws when server returns status error', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'error', error: 'CUDA OOM' })

    await expect(modelController.load('huge')).rejects.toThrow('CUDA OOM')
  })
})
