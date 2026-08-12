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

describe('modelController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GET /models and returns models from API', async () => {
    apiClient.apiGet.mockResolvedValue({
      models: [
        { id: 'm1', name: 'Alpha', type: 'huggingface', size_mb: 512, parameters: 512000000 },
        { name: 'unnamed-id', size_mb: 1.5 },
      ],
    })

    const rows = await modelController.list()

    expect(apiClient.apiGet).toHaveBeenCalledWith('/models/hf')
    expect(rows).toMatchObject([
      { id: 'm1', name: 'Alpha', type: 'huggingface', size_mb: 512 },
      { name: 'unnamed-id' },
    ])
    expect(rows).toHaveLength(2)
  })

  it('handles missing models array', async () => {
    apiClient.apiGet.mockResolvedValue({})

    const rows = await modelController.list()
    expect(rows).toEqual([])
  })

  it('maps description and string-only tags', async () => {
    apiClient.apiGet.mockResolvedValue({
      models: [
        { id: 't1', name: 'Tagged', type: 'local', size_mb: 8, description: 'A model', tags: ['gen', 'small'] },
      ],
    })

    const rows = await modelController.list()
    expect(rows[0]).toMatchObject({
      id: 't1', description: 'A model', tags: ['gen', 'small'], type: 'local',
    })
  })

  it('returns empty array on API error', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('503'))

    const rows = await modelController.list()
    expect(rows).toEqual([])
  })

  it('maps parameters field', async () => {
    apiClient.apiGet.mockResolvedValue({
      models: [{ id: 'p1', name: 'ParamModel', parameters: 1000000000 }],
    })
    const rows = await modelController.list()
    expect(rows[0].parameters).toBe(1000000000)
  })
})
