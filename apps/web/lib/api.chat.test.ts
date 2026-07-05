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

function sseResponse(chunks: string[]) {
  return new ReadableStream({
    start(controller) {
      const enc = new TextEncoder()
      for (const c of chunks) {
        controller.enqueue(enc.encode(c))
      }
      controller.close()
    },
  })
}

import { chatController } from './chat-controller'

describe('chatController.stream', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('POSTs /chat/stream with message and generation fields', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: true, model_type: 'gpt2' })

    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse(['data: {"stream":"chat","phase":"STREAMING","status":"complete","data":{"token":"ok"}}\n\n']),
    } as Response)

    const tokens: string[] = []
    for await (const token of chatController.stream('hi', { max_tokens: 40, temperature: 0.5 })) {
      tokens.push(token)
    }

    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/chat/stream'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          messages: [{ role: 'user', content: 'hi' }],
          max_new_tokens: 40,
          temperature: 0.5,
        }),
      }),
    )
  })
})

describe('chatController.send', () => {
  it('rejects when no model is loaded', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: false, model_type: null })

    await expect(chatController.send('hello')).rejects.toThrow('No model loaded')
  })
})
