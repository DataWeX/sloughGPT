import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))

vi.mock('@/lib/auth', () => ({
  useAuthStore: { getState: () => ({ token: null }) },
}))

vi.mock('@/lib/dev-log', () => ({
  trackEvent: () => {},
}))

function mockFetch(responses: Array<{ ok?: boolean; status?: number; statusText?: string; body?: ReadableStream }>) {
  let callIndex = 0
  return vi.fn().mockImplementation(() => {
    const res = responses[Math.min(callIndex++, responses.length - 1)]
    return Promise.resolve({
      ok: res.ok ?? true,
      status: res.status ?? 200,
      statusText: res.statusText ?? 'OK',
      body: res.body ?? null,
    })
  })
}

function createMockBody(chunks: string[]) {
  let index = 0
  return new ReadableStream({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(new TextEncoder().encode(chunks[index++]))
      } else {
        controller.close()
      }
    },
  })
}

import { createSSEStream, type SSEEnvelope } from './sse-client'

describe('createSSEStream', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('parses SSE envelopes from stream', async () => {
    const event: SSEEnvelope = {
      stream: 'errors',
      phase: 'ERROR',
      status: 'error',
      data: { message: 'fail' },
      meta: {},
      message: 'fail',
    }
    const body = createMockBody([`data: ${JSON.stringify(event)}\n\n`])
    vi.stubGlobal('fetch', mockFetch([{ body }]))

    const onEvent = vi.fn()
    const stream = createSSEStream({ url: '/errors/stream', onEvent })
    stream.start()

    await vi.waitFor(() => expect(onEvent).toHaveBeenCalled())
    expect(onEvent).toHaveBeenCalledWith(event)
    stream.stop()
  })

  it('calls onOpen when connected', async () => {
    const body = createMockBody([])
    vi.stubGlobal('fetch', mockFetch([{ body }]))

    const onOpen = vi.fn()
    const stream = createSSEStream({ url: '/test', onEvent: () => {}, onOpen })
    stream.start()

    await vi.waitFor(() => expect(onOpen).toHaveBeenCalled())
    stream.stop()
  })

  it('calls onClose when stream ends', async () => {
    const body = createMockBody([])
    vi.stubGlobal('fetch', mockFetch([{ body }]))

    const onClose = vi.fn()
    const stream = createSSEStream({ url: '/test', onEvent: () => {}, onClose })
    stream.start()

    await vi.waitFor(() => expect(onClose).toHaveBeenCalled())
  })

  it('calls onError on fetch failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Network')))

    const onError = vi.fn()
    const stream = createSSEStream({ url: '/test', onEvent: () => {}, onError, reconnect: false })
    stream.start()

    await vi.waitFor(() => expect(onError).toHaveBeenCalled())
    expect(onError.mock.calls[0][0].message).toBe('Network')
  })

  it('skips malformed JSON lines', async () => {
    const body = createMockBody(['data: {bad json\n', 'data: {"stream":"ok","phase":"p","status":"success","data":{},"meta":{},"message":"m"}\n\n'])
    vi.stubGlobal('fetch', mockFetch([{ body }]))

    const onEvent = vi.fn()
    const stream = createSSEStream({ url: '/test', onEvent })
    stream.start()

    await vi.waitFor(() => expect(onEvent).toHaveBeenCalledTimes(1))
    expect(onEvent.mock.calls[0][0].stream).toBe('ok')
    stream.stop()
  })

  it('skips empty data lines', async () => {
    const body = createMockBody(['data: \n\n', 'data: {"stream":"x","phase":"p","status":"success","data":{},"meta":{},"message":"m"}\n\n'])
    vi.stubGlobal('fetch', mockFetch([{ body }]))

    const onEvent = vi.fn()
    const stream = createSSEStream({ url: '/test', onEvent })
    stream.start()

    await vi.waitFor(() => expect(onEvent).toHaveBeenCalledTimes(1))
    stream.stop()
  })

  it('stop() aborts in-flight request', async () => {
    const controller = new AbortController()
    const body = createMockBody([])
    const fetchFn = vi.fn().mockImplementation((_url: string, opts: any) => {
      return new Promise((resolve, reject) => {
        opts.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
        setTimeout(() => resolve({ ok: true, status: 200, body }), 10000)
      })
    })
    vi.stubGlobal('fetch', fetchFn)

    const onClose = vi.fn()
    const stream = createSSEStream({ url: '/test', onEvent: () => {}, onClose })
    stream.start()

    await vi.waitFor(() => expect(fetchFn).toHaveBeenCalled())
    stream.stop()
    expect(stream.connected).toBe(false)
  })

  it('reconnects after failure when reconnect=true', async () => {
    let callCount = 0
    vi.stubGlobal('fetch', vi.fn().mockImplementation(() => {
      callCount++
      if (callCount <= 2) return Promise.reject(new Error('fail'))
      const body = createMockBody([])
      return Promise.resolve({ ok: true, status: 200, body })
    }))

    const onError = vi.fn()
    const onOpen = vi.fn()
    const stream = createSSEStream({
      url: '/test',
      onEvent: () => {},
      onError,
      onOpen,
      reconnect: true,
      baseReconnectMs: 100,
      maxReconnectMs: 100,
    })
    stream.start()

    await vi.waitFor(() => expect(onOpen).toHaveBeenCalled())
    expect(callCount).toBeGreaterThanOrEqual(3)
    stream.stop()
  })

  it('does not reconnect when reconnect=false', async () => {
    let callCount = 0
    vi.stubGlobal('fetch', vi.fn().mockImplementation(() => {
      callCount++
      return Promise.reject(new Error('fail'))
    }))

    const onError = vi.fn()
    const stream = createSSEStream({
      url: '/test',
      onEvent: () => {},
      onError,
      reconnect: false,
    })
    stream.start()

    await vi.waitFor(() => expect(onError).toHaveBeenCalled())
    await vi.advanceTimersByTimeAsync(5000)
    expect(callCount).toBe(1)
    stream.stop()
  })

  it('respects maxReconnects limit', async () => {
    let callCount = 0
    vi.stubGlobal('fetch', vi.fn().mockImplementation(() => {
      callCount++
      return Promise.reject(new Error('fail'))
    }))

    const onClose = vi.fn()
    const stream = createSSEStream({
      url: '/test',
      onEvent: () => {},
      onClose,
      reconnect: true,
      maxReconnects: 2,
      baseReconnectMs: 10,
      maxReconnectMs: 10,
    })
    stream.start()

    await vi.waitFor(() => expect(callCount).toBeGreaterThanOrEqual(3))
    await vi.advanceTimersByTimeAsync(100)
    expect(callCount).toBe(3) // 1 initial + 2 reconnects
    expect(onClose).toHaveBeenCalled()
  })

  it('returns false for connected before start', () => {
    const stream = createSSEStream({ url: '/test', onEvent: () => {} })
    expect(stream.connected).toBe(false)
  })

  it('skips non-data lines', async () => {
    const body = createMockBody([
      'event: message\n',
      'id: 1\n',
      'retry: 3000\n',
      'data: {"stream":"x","phase":"p","status":"success","data":{},"meta":{},"message":"m"}\n\n',
    ])
    vi.stubGlobal('fetch', mockFetch([{ body }]))

    const onEvent = vi.fn()
    const stream = createSSEStream({ url: '/test', onEvent })
    stream.start()

    await vi.waitFor(() => expect(onEvent).toHaveBeenCalledTimes(1))
    stream.stop()
  })

  it('handles multiple events in one chunk', async () => {
    const e1: SSEEnvelope = { stream: 'a', phase: 'p', status: 'success', data: {}, meta: {}, message: '1' }
    const e2: SSEEnvelope = { stream: 'b', phase: 'p', status: 'success', data: {}, meta: {}, message: '2' }
    const body = createMockBody([`data: ${JSON.stringify(e1)}\n\ndata: ${JSON.stringify(e2)}\n\n`])
    vi.stubGlobal('fetch', mockFetch([{ body }]))

    const onEvent = vi.fn()
    const stream = createSSEStream({ url: '/test', onEvent })
    stream.start()

    await vi.waitFor(() => expect(onEvent).toHaveBeenCalledTimes(2))
    stream.stop()
  })

  it('handles split chunks (partial line)', async () => {
    const event: SSEEnvelope = { stream: 'x', phase: 'p', status: 'success', data: {}, meta: {}, message: 'm' }
    const json = `data: ${JSON.stringify(event)}\n\n`
    const mid = Math.floor(json.length / 2)
    const body = createMockBody([json.slice(0, mid), json.slice(mid)])
    vi.stubGlobal('fetch', mockFetch([{ body }]))

    const onEvent = vi.fn()
    const stream = createSSEStream({ url: '/test', onEvent })
    stream.start()

    await vi.waitFor(() => expect(onEvent).toHaveBeenCalledTimes(1))
    stream.stop()
  })
})
