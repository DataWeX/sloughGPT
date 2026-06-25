import { beforeEach, describe, expect, it, vi } from 'vitest'

const mockStatus = vi.hoisted(() => vi.fn().mockResolvedValue({ loaded: true, model_type: 'gpt2', device: 'cpu' }))

vi.mock('./model-controller', () => ({
  modelController: { status: mockStatus },
}))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { chatController } from './chat-controller'

describe('chatController', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockStatus.mockResolvedValue({ loaded: true, model_type: 'gpt2', device: 'cpu' })
  })

  describe('checkReady', () => {
    it('returns model status', async () => {
      const result = await chatController.checkReady()
      expect(result).toEqual({ loaded: true, model_type: 'gpt2', device: 'cpu' })
    })
  })

  describe('saveSessionContext', () => {
    it('POSTs to /session/{id}/context', async () => {
      apiClient.apiPost.mockResolvedValue({ status: 'ok' })
      await chatController.saveSessionContext('sess-1', [{ role: 'user', content: 'hi' }])
      expect(apiClient.apiPost).toHaveBeenCalledWith('/session/sess-1/context', {
        messages: [{ role: 'user', content: 'hi' }],
      })
    })
  })

  describe('formatMessages', () => {
    it('formats user message', () => {
      const result = chatController.formatMessages([{ role: 'user', content: 'Hello' }])
      expect(result).toBe('User: Hello\nAssistant:')
    })

    it('formats assistant message', () => {
      const result = chatController.formatMessages([{ role: 'assistant', content: 'Hi!' }])
      expect(result).toBe('Assistant: Hi!\nAssistant:')
    })

    it('formats system message', () => {
      const result = chatController.formatMessages([{ role: 'system', content: 'Be helpful' }])
      expect(result).toBe('System: Be helpful\nAssistant:')
    })

    it('formats multi-turn conversation', () => {
      const result = chatController.formatMessages([
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi!' },
        { role: 'user', content: 'How are you?' },
      ])
      expect(result).toBe('User: Hello\nAssistant: Hi!\nUser: How are you?\nAssistant:')
    })
  })

  describe('send', () => {
    it('throws if no model loaded', async () => {
      mockStatus.mockResolvedValue({ loaded: false, model_type: null, device: null })
      await expect(chatController.send('hi')).rejects.toThrow('No model loaded')
    })

    it('POSTs to /chat and returns response', async () => {
      apiClient.apiPost.mockResolvedValue({ message: 'Hello!', session_id: 's1' })
      const result = await chatController.send('hi')
      expect(result.message).toBe('Hello!')
      expect(result.session_id).toBe('s1')
      expect(apiClient.apiPost).toHaveBeenCalledWith('/chat', {
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 100,
        temperature: 0.8,
      })
    })

    it('falls back to /inference/generate if /chat fails', async () => {
      apiClient.apiPost.mockRejectedValueOnce(new Error('fail'))
      apiClient.apiPost.mockResolvedValue({ text: 'fallback response' })
      const result = await chatController.send('hi')
      expect(result.message).toBe('fallback response')
    })
  })
})
