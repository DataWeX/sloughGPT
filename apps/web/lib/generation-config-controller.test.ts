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

import { generationConfigController } from './generation-config-controller'

describe('generationConfigController.get', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /config/generation', async () => {
    apiClient.apiGet.mockResolvedValue({ temperature: 0.8, max_new_tokens: 100, top_p: 0.9, top_k: 40 })

    const result = await generationConfigController.get()
    expect(result.temperature).toBe(0.8)
    expect(result.max_new_tokens).toBe(100)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/config/generation')
  })

  it('returns all config fields', async () => {
    apiClient.apiGet.mockResolvedValue({ temperature: 0.5, max_new_tokens: 200, top_p: 0.95, top_k: 50 })
    const result = await generationConfigController.get()
    expect(result.temperature).toBe(0.5)
    expect(result.max_new_tokens).toBe(200)
    expect(result.top_p).toBe(0.95)
    expect(result.top_k).toBe(50)
  })
})

describe('generationConfigController.update', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('PATCHes /config/generation', async () => {
    apiClient.apiPatch.mockResolvedValue(undefined)

    await generationConfigController.update({ temperature: 0.7 })
    expect(apiClient.apiPatch).toHaveBeenCalledWith('/config/generation', { temperature: 0.7 })
  })

  it('sends multiple fields', async () => {
    apiClient.apiPatch.mockResolvedValue(undefined)
    await generationConfigController.update({ temperature: 0.6, max_new_tokens: 150 })
    expect(apiClient.apiPatch).toHaveBeenCalledWith('/config/generation', { temperature: 0.6, max_new_tokens: 150 })
  })

  it('sends empty object for no changes', async () => {
    apiClient.apiPatch.mockResolvedValue(undefined)
    await generationConfigController.update({})
    expect(apiClient.apiPatch).toHaveBeenCalledWith('/config/generation', {})
  })
})
