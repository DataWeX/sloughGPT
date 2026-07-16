import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

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

import { generateController } from './generate-controller'

describe('generateController.generateStream', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('fetch', vi.fn())
  })

  it('invokes onToken for each data frame and onDone when stream completes', async () => {
    const tokens: string[] = []
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse([
        'data: {"stream":"generate","phase":"STREAMING","status":"working","data":{"token":"Hel"}}\n\n',
        'data: {"stream":"generate","phase":"STREAMING","status":"complete","data":{"token":"lo"}}\n\n',
      ]),
    } as Response)

    await new Promise<void>((resolve) => {
      generateController.generateStream({ prompt: 'hi' }, (t) => tokens.push(t), () => resolve())
    })

    expect(tokens).toEqual(['Hel', 'lo'])
  })

  it('POSTs /inference/generate/stream with inference payload', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse(['data: {"stream":"generate","phase":"STREAMING","status":"complete","data":{"token":"x"}}\n\n']),
    } as Response)

    await new Promise<void>((resolve) => {
      generateController.generateStream(
        { prompt: 'p', max_new_tokens: 42, temperature: 0.5, top_p: 0.7, top_k: 10 },
        () => {},
        () => resolve(),
      )
    })

    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:9/inference/generate/stream',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          prompt: 'p',
          max_new_tokens: 42,
          temperature: 0.5,
          top_p: 0.7,
          top_k: 10,
        }),
      }),
    )
  })

  it('calls onDone without tokens when response is not ok', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 503,
      body: null,
    } as Response)

    const tokens: string[] = []
    await new Promise<void>((resolve) => {
      generateController.generateStream({ prompt: 'x' }, (t) => tokens.push(t), () => resolve())
    })
    expect(tokens).toEqual([])
  })

  it('stops on SSE error payload', async () => {
    const tokens: string[] = []
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse([
        'data: {"stream":"generate","phase":"STREAMING","status":"working","data":{"token":"a"}}\n\n',
        'data: {"stream":"generate","phase":"STREAMING","status":"error","data":{"error":"boom"}}\n\n',
        'data: {"stream":"generate","phase":"STREAMING","status":"working","data":{"token":"b"}}\n\n',
      ]),
    } as Response)

    await new Promise<void>((resolve) => {
      generateController.generateStream({ prompt: 'x' }, (t) => tokens.push(t), () => resolve())
    })

    expect(tokens).toEqual(['a'])
  })
})
