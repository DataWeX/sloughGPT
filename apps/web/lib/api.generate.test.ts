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

import { generateController } from './generate-controller'

describe('generateController.generate', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('throws when server returns 200 with error field (no model loaded)', async () => {
    apiClient.apiPost.mockResolvedValue({ error: 'Model not loaded', text: '' })

    await expect(generateController.generate({ prompt: 'hello' })).rejects.toThrow('Model not loaded')
  })

  it('returns text when server returns a non-empty body', async () => {
    apiClient.apiPost.mockResolvedValue({ text: 'hi there', model: 'gpt2-engine', tokens_generated: 2 })

    const out = await generateController.generate({ prompt: 'hello' })
    expect(out.text).toBe('hi there')
    expect(out.model).toBe('gpt2-engine')
    expect(out.tokens_generated).toBe(2)
  })

  it('throws on HTTP error responses', async () => {
    apiClient.apiPost.mockRejectedValue(new Error('503'))

    await expect(generateController.generate({ prompt: 'x' })).rejects.toThrow('503')
  })
})