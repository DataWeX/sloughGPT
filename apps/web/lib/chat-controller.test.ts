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

  it('POSTs /chat with message and generation fields', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: true, model_type: 'gpt2' })

    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse([
        'data: {"stream":"chat","phase":"STREAMING","status":"complete","data":{"token":"ok"}}\n\n',
      ]),
    } as Response)

    const tokens: string[] = []
    for await (const token of chatController.stream('hi', { max_tokens: 40, temperature: 0.5 })) {
      tokens.push(token)
    }

    expect(fetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/chat$/),
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

  it('yields error when model not loaded', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: false, model_type: null })
    const tokens: string[] = []
    for await (const token of chatController.stream('hi')) {
      tokens.push(token)
    }
    expect(tokens).toContain('[No model loaded]')
  })

  it('yields tokens from multiple SSE events', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: true, model_type: 'gpt2' })
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse([
        'data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":"Hello"}}\n\n',
        'data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":" world"}}\n\n',
        'data: {"stream":"chat","phase":"STREAMING","status":"complete","data":{}}\n\n',
      ]),
    } as Response)

    const tokens: string[] = []
    for await (const token of chatController.stream('hi')) {
      tokens.push(token)
    }
    expect(tokens).toEqual(['Hello', ' world'])
  })

  it('yields error on SSE error event', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: true, model_type: 'gpt2' })
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse([
        'data: {"stream":"chat","phase":"STREAMING","status":"error","message":"server error"}\n\n',
      ]),
    } as Response)

    const tokens: string[] = []
    for await (const token of chatController.stream('hi')) {
      tokens.push(token)
    }
    expect(tokens).toEqual(['[server error]'])
  })

  it('yields connection error on fetch failure', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: true, model_type: 'gpt2' })
    vi.mocked(fetch).mockRejectedValue(new Error('network fail'))

    const tokens: string[] = []
    for await (const token of chatController.stream('hi')) {
      tokens.push(token)
    }
    expect(tokens[0]).toContain('[Connection error:')
    expect(tokens[0]).toContain('network fail')
  })

  it('collects lifecycle frames and stops on the finalized terminal', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: true, model_type: 'gpt2' })
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse([
        'data: {"stream":"chat","phase":"PROCESSING","status":"working","data":{}}\n\n',
        'data: {"stream":"chat","phase":"PROCESSING","status":"generating","data":{}}\n\n',
        'data: {"stream":"chat","phase":"STREAMING","status":"streaming","data":{"token":"Hi"}}\n\n',
        'data: {"stream":"chat","phase":"PROCESSING","status":"finalizing","data":{}}\n\n',
        'data: {"stream":"chat","phase":"FINALIZED","status":"finalized","data":{},"meta":{"usage_tokens":1}}\n\n',
      ]),
    } as Response)

    const tokens: string[] = []
    for await (const token of chatController.stream('hi')) {
      tokens.push(token)
    }
    expect(tokens).toEqual(['Hi'])
  })

  it('stops on the cancelled terminal without an error', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: true, model_type: 'gpt2' })
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse([
        'data: {"stream":"chat","phase":"STREAMING","status":"streaming","data":{"token":"half"}}\n\n',
        'data: {"stream":"chat","phase":"CANCELLED","status":"cancelled","data":{}}\n\n',
      ]),
    } as Response)

    const tokens: string[] = []
    for await (const token of chatController.stream('hi')) {
      tokens.push(token)
    }
    expect(tokens).toEqual(['half'])
  })
})

describe('chatController.formatMessages', () => {
  it('formats user and assistant messages', () => {
    const result = chatController.formatMessages([
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'hi' },
    ])
    expect(result).toBe('User: hello\nAssistant: hi\nAssistant:')
  })

  it('formats system messages', () => {
    const result = chatController.formatMessages([
      { role: 'system', content: 'be helpful' },
      { role: 'user', content: 'hi' },
    ])
    expect(result).toBe('System: be helpful\nUser: hi\nAssistant:')
  })

  it('handles empty messages', () => {
    const result = chatController.formatMessages([])
    expect(result).toBe('Assistant:')
  })
})

describe('chatController.checkReady', () => {
  it('returns model status', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: true, model_type: 'gpt2' })
    const status = await chatController.checkReady()
    expect(status.loaded).toBe(true)
  })
})

describe('chatController.getSuggestions', () => {
  it('returns suggestions on success', async () => {
    apiClient.apiGet.mockResolvedValue({ suggestions: [{ text: 'Try this', icon: 'bulb' }] })
    const suggestions = await chatController.getSuggestions()
    expect(suggestions).toEqual([{ text: 'Try this', icon: 'bulb' }])
  })

  it('returns empty array on failure', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('fail'))
    const suggestions = await chatController.getSuggestions()
    expect(suggestions).toEqual([])
  })

  it('returns empty array when suggestions field missing', async () => {
    apiClient.apiGet.mockResolvedValue({})
    const suggestions = await chatController.getSuggestions()
    expect(suggestions).toEqual([])
  })
})

describe('chatController.inspectContext', () => {
  it('GETs /context/inspect and returns the inspector', async () => {
    const inspector = { system_prompt: 'be warm', working_memory: [], semantic_keys: ['espresso'] }
    apiClient.apiGet.mockResolvedValue(inspector)
    const result = await chatController.inspectContext()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/context/inspect')
    expect(result).toEqual(inspector)
  })

  it('returns null when the backend is unavailable', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('down'))
    const result = await chatController.inspectContext()
    expect(result).toBeNull()
  })
})

describe('chatController.saveSessionContext', () => {
  it('calls apiPost with session context', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'stored' })
    await chatController.saveSessionContext('sess-1', [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
    ])
    expect(apiClient.apiPost).toHaveBeenCalledWith('/session/sess-1/context', {
      messages: [
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: 'hello' },
      ],
    })
  })
})

describe('chatController.regenerateStream', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('yields tokens from regenerate endpoint', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse([
        'data: {"stream":"chat","phase":"STREAMING","status":"streaming","data":{"token":"re"}}\n\n',
        'data: {"stream":"chat","phase":"FINALIZED","status":"finalized","data":{}}\n\n',
      ]),
    } as Response)

    const results: any[] = []
    for await (const r of chatController.regenerateStream('sess-1', [
      { role: 'user', content: 'hi' },
    ])) {
      results.push(r)
    }
    expect(results).toEqual([{ token: 're' }, { done: true }])
  })

  it('yields done on legacy complete frame (grandfathered tolerance)', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse([
        'data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":"re"}}\n\n',
        'data: {"stream":"chat","phase":"STREAMING","status":"complete","data":{}}\n\n',
      ]),
    } as Response)

    const results: any[] = []
    for await (const r of chatController.regenerateStream('sess-1', [])) {
      results.push(r)
    }
    expect(results).toEqual([{ token: 're' }, { done: true }])
  })

  it('yields error on SSE error event', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse(['data: {"stream":"chat","status":"error","message":"regen failed"}\n\n']),
    } as Response)

    const results: any[] = []
    for await (const r of chatController.regenerateStream('sess-1', [])) {
      results.push(r)
    }
    expect(results).toEqual([{ error: 'regen failed' }])
  })

  it('yields connection error on fetch failure', async () => {
    vi.mocked(fetch).mockRejectedValue(new Error('connection lost'))

    const results: any[] = []
    for await (const r of chatController.regenerateStream('sess-1', [])) {
      results.push(r)
    }
    expect(results[0].error).toContain('Connection error')
    expect(results[0].error).toContain('connection lost')
  })
})

describe('chatController control methods', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('cancelStream sends cancel control', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', data: { cancelled: true } })
    await chatController.cancelStream('session-1')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/chat/control', {
      session_id: 'session-1',
      action: 'cancel',
    })
  })

  it('approveTool sends approve control', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', data: { stored: true } })
    await chatController.approveTool('session-1', 'calculator', true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/chat/control', {
      session_id: 'session-1',
      action: 'approve',
      tool_name: 'calculator',
      approved: true,
    })
  })

  it('injectContext sends context control', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', data: { stored: true } })
    await chatController.injectContext('session-1', 'extra context')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/chat/control', {
      session_id: 'session-1',
      action: 'context',
      context: 'extra context',
    })
  })
})
