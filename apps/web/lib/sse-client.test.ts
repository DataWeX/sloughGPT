import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import type { SSEStream } from './sse-client'

const { mockFetch } = vi.hoisted(() => ({ mockFetch: vi.fn() }))

vi.mock('./config', () => ({ PUBLIC_API_URL: 'http://api.test' }))

const { mockToken } = vi.hoisted(() => ({
  mockToken: vi.fn<() => string | null>(() => null),
}))
vi.mock('./auth', () => ({
  useAuthStore: { getState: () => ({ token: mockToken() }) },
}))

import { createSSEStream } from './sse-client'

function makeResponse(chunks: string[], ok = true, status = 200) {
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c))
      controller.close()
    },
  })
  return {
    ok,
    status,
    statusText: ok ? 'OK' : 'Nope',
    body: { getReader: () => stream.getReader() },
  }
}

const ENV_ONE = 'data: {"stream":"health","phase":"IDLE","status":"working","data":{},"meta":{},"message":"hi"}\n\n'
const ENV_TWO = 'data: {"stream":"health","phase":"IDLE","status":"complete","data":{},"meta":{},"message":"done"}\n\n'

describe('createSSEStream', () => {
  beforeEach(() => {
    mockFetch.mockReset()
    mockToken.mockReturnValue(null)
    vi.stubGlobal('fetch', mockFetch)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('connects to PUBLIC_API_URL + url and parses envelopes', async () => {
    mockFetch.mockResolvedValueOnce(makeResponse([ENV_ONE, ENV_TWO]))
    const onEvent = vi.fn()
    const onOpen = vi.fn()
    const onClose = vi.fn()
    const stream = createSSEStream({ url: '/health/stream', onEvent, onOpen, onClose, reconnect: false })
    stream.start()

    await vi.waitFor(() => expect(onClose).toHaveBeenCalled())
    expect(mockFetch).toHaveBeenCalledWith(
      'http://api.test/health/stream',
      expect.objectContaining({ headers: expect.objectContaining({ Accept: 'text/event-stream' }) }),
    )
    expect(onOpen).toHaveBeenCalledTimes(1)
    expect(onEvent).toHaveBeenCalledTimes(2)
    expect(onEvent).toHaveBeenNthCalledWith(1, expect.objectContaining({ message: 'hi', status: 'working' }))
    expect(onEvent).toHaveBeenNthCalledWith(2, expect.objectContaining({ message: 'done', status: 'complete' }))
    expect(stream.connected).toBe(false)
  })

  it('attaches the bearer token when the auth store has one', async () => {
    mockToken.mockReturnValue('sekret')
    mockFetch.mockResolvedValueOnce(makeResponse([]))
    createSSEStream({ url: '/auth-stream', onEvent: () => {} }).start()

    await vi.waitFor(() => expect(mockFetch).toHaveBeenCalled())
    const [, init] = mockFetch.mock.calls[0] as [string, { headers: Record<string, string> }]
    expect(init.headers.Authorization).toBe('Bearer sekret')
  })

  it('skips non-data lines and malformed JSON', async () => {
    mockFetch.mockResolvedValueOnce(makeResponse(['event: ping\n\n', 'data: not-json\n\n', ENV_ONE]))
    const onEvent = vi.fn()
    createSSEStream({ url: '/noisy', onEvent, reconnect: false }).start()
    await vi.waitFor(() => expect(onEvent).toHaveBeenCalled())
    expect(onEvent).toHaveBeenCalledTimes(1)
  })

  it('buffers envelopes split across read chunks', async () => {
    mockFetch.mockResolvedValueOnce(makeResponse([
      'data: {"stream":"health","data":{"a":1},"meta":{"x":',
      '2},"message":"m"}\n\n',
      ENV_TWO,
    ]))
    const onEvent = vi.fn()
    createSSEStream({ url: '/chunked', onEvent, reconnect: false }).start()
    await vi.waitFor(() => expect(onEvent).toHaveBeenCalledTimes(2))
    expect(onEvent).toHaveBeenNthCalledWith(1, expect.objectContaining({ data: { a: 1 }, meta: { x: 2 } }))
  })

  it('reports connected true while the stream is open', async () => {
    mockFetch.mockResolvedValueOnce(makeResponse([ENV_ONE]))
    let connectedDuringOpen = false
    let ref: SSEStream | null = null
    ref = createSSEStream({
      url: '/open',
      onEvent: () => {},
      onOpen: () => {
        connectedDuringOpen = ref!.connected
      },
      reconnect: false,
    })
    ref.start()
    await vi.waitFor(() => expect(connectedDuringOpen).toBe(true))
    ref.stop()
  })

  it('reports an error and closes when the response is not ok', async () => {
    mockFetch.mockResolvedValueOnce(makeResponse([], false, 500))
    const onError = vi.fn()
    const onClose = vi.fn()
    createSSEStream({ url: '/fail', onEvent: () => {}, onError, onClose, reconnect: false }).start()
    await vi.waitFor(() => expect(onClose).toHaveBeenCalled())
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: 'SSE 500: Nope' }))
  })

  it('reconnects after a network failure and recovers', async () => {
    mockFetch.mockRejectedValueOnce(new Error('network down'))
    mockFetch.mockResolvedValueOnce(makeResponse([ENV_ONE]))
    const onEvent = vi.fn()
    const onError = vi.fn()
    const onClose = vi.fn()
    createSSEStream({
      url: '/recover',
      onEvent,
      onError,
      onClose,
      baseReconnectMs: 10,
      maxReconnects: 3,
    }).start()

    await vi.waitFor(() => expect(onEvent).toHaveBeenCalled())
    expect(onError).toHaveBeenCalledTimes(1)
    expect(mockFetch).toHaveBeenCalledTimes(2)
  })

  it('stops reconnecting after maxReconnects is exhausted', async () => {
    mockFetch.mockRejectedValue(new Error('down'))
    const onError = vi.fn()
    const onClose = vi.fn()
    createSSEStream({
      url: '/exhaust',
      onEvent: () => {},
      onError,
      onClose,
      baseReconnectMs: 10,
      maxReconnects: 2,
    }).start()

    await vi.waitFor(() => expect(onClose).toHaveBeenCalled())
    expect(mockFetch).toHaveBeenCalledTimes(3)
    expect(onError).toHaveBeenCalledTimes(3)
  })

  it('aborts on stop() and does not reconnect', async () => {
    mockFetch.mockImplementation((_url: string, init?: { signal?: AbortSignal }) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () =>
          reject(Object.assign(new Error('Aborted'), { name: 'AbortError' })),
        )
      }),
    )
    const onError = vi.fn()
    const onClose = vi.fn()
    const stream = createSSEStream({
      url: '/abort',
      onEvent: () => {},
      onError,
      onClose,
      baseReconnectMs: 10,
      maxReconnects: 3,
    })
    stream.start()
    stream.stop()

    await vi.waitFor(() => expect(onClose).toHaveBeenCalled())
    expect(onError).not.toHaveBeenCalled()
    expect(mockFetch).toHaveBeenCalledTimes(1)
    expect(stream.connected).toBe(false)
  })
})
