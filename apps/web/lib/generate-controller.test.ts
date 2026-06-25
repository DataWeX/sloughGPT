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

  it('POSTs /inference/generate with request body', async () => {
    apiClient.apiPost.mockResolvedValue({ text: 'Hello world', model: 'gpt2', tokens_generated: 3 })
    const result = await generateController.generate({ prompt: 'Hi' })
    expect(result.text).toBe('Hello world')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/inference/generate', { prompt: 'Hi' })
  })

  it('throws on error response', async () => {
    apiClient.apiPost.mockResolvedValue({ error: 'Model not loaded', text: '', tokens_generated: 0 })
    await expect(generateController.generate({ prompt: 'Hi' })).rejects.toThrow('Model not loaded')
  })

  it('passes all optional fields', async () => {
    apiClient.apiPost.mockResolvedValue({ text: 'ok' })
    await generateController.generate({ prompt: 'Hi', max_new_tokens: 100, temperature: 0.8, top_p: 0.9, top_k: 40, repetition_penalty: 1.1, model: 'gpt2' })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/inference/generate', {
      prompt: 'Hi', max_new_tokens: 100, temperature: 0.8, top_p: 0.9, top_k: 40, repetition_penalty: 1.1, model: 'gpt2',
    })
  })
})
