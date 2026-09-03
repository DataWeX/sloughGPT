import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockGetState = vi.fn().mockReturnValue({ token: null as string | null })

vi.mock('./auth', () => ({
  useAuthStore: {
    getState: (...args: any[]) => mockGetState(...args),
  },
}))

vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

vi.mock('./error-store', () => ({
  useErrorStore: { getState: () => ({ addError: vi.fn() }) },
}))

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

import {
  ApiError, apiGet, apiPost, apiPut, apiDelete, apiPatch, createApiClient,
  InterceptorManager, HttpCache, CircuitBreaker, Throttler,
  httpClient, createHttpClient,
  type RequestConfig, type ResponseEnvelope,
} from './http-client'

function mockOk(body: unknown = { data: 'test' }) {
  return {
    ok: true,
    status: 200,
    text: () => Promise.resolve(JSON.stringify(body)),
    headers: { get: () => null },
  }
}

function mockError(status: number, body: string = '{"detail":"err"}') {
  return {
    ok: false,
    status,
    statusText: 'Error',
    text: () => Promise.resolve(body),
    headers: { get: () => null },
  }
}

describe('ApiError', () => {
  it('constructor sets message, status, data, name', () => {
    const err = new ApiError('fail', 404, { raw: 'x' }, 'req-1')
    expect(err.message).toBe('fail')
    expect(err.status).toBe(404)
    expect(err.data).toEqual({ raw: 'x' })
    expect(err.requestId).toBe('req-1')
    expect(err.name).toBe('ApiError')
    expect(err).toBeInstanceOf(Error)
  })
})

describe('apiGet', () => {
  beforeEach(() => { vi.clearAllMocks(); mockFetch.mockReset(); mockGetState.mockReturnValue({ token: null }) })

  it('calls fetch with GET and correct URL', async () => {
    mockFetch.mockResolvedValue(mockOk())
    await apiGet('/health')
    expect(mockFetch).toHaveBeenCalledTimes(1)
    const [url, init] = mockFetch.mock.calls[0]
    expect(url).toBe('http://127.0.0.1:9/health')
    expect(init.method).toBe('GET')
  })

  it('includes JSON content-type header', async () => {
    mockFetch.mockResolvedValue(mockOk())
    await apiGet('/test')
    const [, init] = mockFetch.mock.calls[0]
    expect(init.headers['Content-Type']).toBe('application/json')
  })

  it('appends query string from params', async () => {
    mockFetch.mockResolvedValue(mockOk())
    await apiGet('/search', { q: 'hello', page: '2' })
    const [url] = mockFetch.mock.calls[0]
    expect(url).toContain('q=hello&page=2')
  })

  it('returns parsed JSON body', async () => {
    mockFetch.mockResolvedValue(mockOk({ result: 42 }))
    const result = await apiGet<{ result: number }>('/test')
    expect(result).toEqual({ result: 42 })
  })

  it('returns undefined for empty response', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve(''),
      headers: { get: () => null },
    })
    const result = await apiGet('/empty')
    expect(result).toBeUndefined()
  })

  it('throws ApiError on non-ok response', async () => {
    mockFetch.mockResolvedValue(mockError(400))
    await expect(apiGet('/bad')).rejects.toThrow(ApiError)
    try {
      await apiGet('/bad')
    } catch (e: any) {
      expect(e.status).toBe(400)
    }
  })

  it('retries on retryable status 503', async () => {
    mockFetch
      .mockResolvedValueOnce(mockError(503))
      .mockResolvedValueOnce(mockOk({ ok: true }))
    const result = await apiGet('/flaky')
    expect(mockFetch).toHaveBeenCalledTimes(2)
    expect(result).toEqual({ ok: true })
  })

  it('retries network error up to 2 times then throws', async () => {
    mockFetch.mockRejectedValue(new Error('Failed to fetch'))
    await expect(apiGet('/down')).rejects.toThrow(ApiError)
    expect(mockFetch).toHaveBeenCalledTimes(3)
  })

  it('does not retry or relabel when the caller-provided signal is aborted', async () => {
    const controller = new AbortController()
    mockFetch.mockRejectedValue(new DOMException('The operation was aborted.', 'AbortError'))
    controller.abort()

    await expect(apiGet('/cancelled', undefined, { signal: controller.signal })).rejects.toThrow(DOMException)

    expect(mockFetch).toHaveBeenCalledTimes(1)
  })

  it('retries on network error then succeeds', async () => {
    mockFetch
      .mockRejectedValueOnce(new Error('Failed to fetch'))
      .mockResolvedValueOnce(mockOk({ ok: true }))
    const result = await apiGet('/flaky-net')
    expect(result).toEqual({ ok: true })
    expect(mockFetch).toHaveBeenCalledTimes(2)
  })

  it('includes Authorization header when token is set', async () => {
    mockGetState.mockReturnValue({ token: 'abc123' })
    mockFetch.mockResolvedValue(mockOk())
    await apiGet('/auth')
    const [, init] = mockFetch.mock.calls[0]
    expect(init.headers['Authorization']).toBe('Bearer abc123')
    mockGetState.mockReturnValue({ token: null })
  })

  it('skips Authorization header when noAuth option is set', async () => {
    mockGetState.mockReturnValue({ token: 'abc123' })
    mockFetch.mockResolvedValue(mockOk())
    await apiGet('/public', undefined, { noAuth: true })
    const [, init] = mockFetch.mock.calls[0]
    expect(init.headers['Authorization']).toBeUndefined()
    mockGetState.mockReturnValue({ token: null })
  })
})

describe('apiPost', () => {
  beforeEach(() => { vi.clearAllMocks(); mockFetch.mockReset() })

  it('calls fetch with POST and stringified body', async () => {
    mockFetch.mockResolvedValue(mockOk({ created: true }))
    await apiPost('/items', { name: 'test' })
    expect(mockFetch).toHaveBeenCalledTimes(1)
    const [url, init] = mockFetch.mock.calls[0]
    expect(url).toBe('http://127.0.0.1:9/items')
    expect(init.method).toBe('POST')
    expect(init.body).toBe(JSON.stringify({ name: 'test' }))
  })
})

describe('apiPut', () => {
  beforeEach(() => { vi.clearAllMocks(); mockFetch.mockReset() })

  it('calls fetch with PUT and body', async () => {
    mockFetch.mockResolvedValue(mockOk())
    await apiPut('/items/1', { name: 'updated' })
    const [, init] = mockFetch.mock.calls[0]
    expect(init.method).toBe('PUT')
    expect(init.body).toBe(JSON.stringify({ name: 'updated' }))
  })
})

describe('apiDelete', () => {
  beforeEach(() => { vi.clearAllMocks(); mockFetch.mockReset() })

  it('calls fetch with DELETE and no body', async () => {
    mockFetch.mockResolvedValue(mockOk())
    await apiDelete('/items/1')
    const [, init] = mockFetch.mock.calls[0]
    expect(init.method).toBe('DELETE')
    expect(init.body).toBeUndefined()
  })
})

describe('apiPatch', () => {
  beforeEach(() => { vi.clearAllMocks(); mockFetch.mockReset() })

  it('calls fetch with PATCH and body', async () => {
    mockFetch.mockResolvedValue(mockOk())
    await apiPatch('/items/1', { name: 'patched' })
    const [, init] = mockFetch.mock.calls[0]
    expect(init.method).toBe('PATCH')
    expect(init.body).toBe(JSON.stringify({ name: 'patched' }))
  })
})

describe('createApiClient', () => {
  beforeEach(() => { vi.clearAllMocks(); mockFetch.mockReset() })

  it('returns client with get/post/put/delete/patch methods', () => {
    const client = createApiClient()
    expect(typeof client.get).toBe('function')
    expect(typeof client.post).toBe('function')
    expect(typeof client.put).toBe('function')
    expect(typeof client.delete).toBe('function')
    expect(typeof client.patch).toBe('function')
  })

  it('defaults baseURL to PUBLIC_API_URL', () => {
    const client = createApiClient()
    expect(client.defaults.baseURL).toBe('http://127.0.0.1:9')
  })

  it('custom baseURL is stored in defaults', () => {
    const client = createApiClient('http://custom:3000')
    expect(client.defaults.baseURL).toBe('http://custom:3000')
  })

  it('client.get passes params to apiGet', async () => {
    const client = createApiClient()
    mockFetch.mockResolvedValue(mockOk())
    await client.get('/data', { params: { page: '1' } })
    const [url] = mockFetch.mock.calls[0]
    expect(url).toContain('page=1')
  })

  it('client.post calls apiPost with body', async () => {
    const client = createApiClient()
    mockFetch.mockResolvedValue(mockOk())
    await client.post('/data', { x: 1 })
    const [, init] = mockFetch.mock.calls[0]
    expect(init.method).toBe('POST')
    expect(init.body).toBe(JSON.stringify({ x: 1 }))
  })

  it('client.put calls apiPut with body', async () => {
    const client = createApiClient()
    mockFetch.mockResolvedValue(mockOk())
    await client.put('/data', { x: 2 })
    const [, init] = mockFetch.mock.calls[0]
    expect(init.method).toBe('PUT')
  })

  it('client.delete calls apiDelete', async () => {
    const client = createApiClient()
    mockFetch.mockResolvedValue(mockOk())
    await client.delete('/data')
    const [, init] = mockFetch.mock.calls[0]
    expect(init.method).toBe('DELETE')
  })

  it('client.patch calls apiPatch with body', async () => {
    const client = createApiClient()
    mockFetch.mockResolvedValue(mockOk())
    await client.patch('/data', { x: 3 })
    const [, init] = mockFetch.mock.calls[0]
    expect(init.method).toBe('PATCH')
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// New feature tests
// ═══════════════════════════════════════════════════════════════════════════

describe('InterceptorManager', () => {
  it('runs interceptors in order', async () => {
    const mgr = new InterceptorManager<string>()
    const order: string[] = []
    mgr.use(async (v) => { order.push('a'); return v + '-a' })
    mgr.use(async (v) => { order.push('b'); return v + '-b' })
    const result = await mgr.run('start')
    expect(result).toBe('start-a-b')
    expect(order).toEqual(['a', 'b'])
  })

  it('eject removes interceptor by id', async () => {
    const mgr = new InterceptorManager<number>()
    const id = mgr.use(async (v) => v + 1)
    mgr.use(async (v) => v + 10)
    mgr.eject(id)
    const result = await mgr.run(0)
    expect(result).toBe(10)
  })

  it('clear removes all interceptors', async () => {
    const mgr = new InterceptorManager<number>()
    mgr.use(async (v) => v + 1)
    mgr.use(async (v) => v + 2)
    mgr.clear()
    expect(mgr.size).toBe(0)
    const result = await mgr.run(0)
    expect(result).toBe(0)
  })

  it('onRejected catches errors from fulfilled', async () => {
    const mgr = new InterceptorManager<string>()
    mgr.use(
      async () => { throw new Error('fail') },
      async () => 'recovered',
    )
    const result = await mgr.run('input')
    expect(result).toBe('recovered')
  })

  it('size returns count of interceptors', () => {
    const mgr = new InterceptorManager<void>()
    expect(mgr.size).toBe(0)
    mgr.use(async () => {})
    expect(mgr.size).toBe(1)
    mgr.use(async () => {})
    expect(mgr.size).toBe(2)
  })
})

describe('HttpCache', () => {
  it('stores and retrieves values', () => {
    const cache = new HttpCache({ ttlMs: 1000 })
    cache.set('key1', { foo: 'bar' })
    const hit = cache.get('key1')
    expect(hit).toBeDefined()
    expect(hit!.data).toEqual({ foo: 'bar' })
    expect(hit!.stale).toBe(false)
  })

  it('returns undefined for cache miss', () => {
    const cache = new HttpCache({ ttlMs: 1000 })
    expect(cache.get('missing')).toBeUndefined()
    expect(cache.stats.misses).toBe(1)
  })

  it('evicts after TTL expires', async () => {
    const cache = new HttpCache({ ttlMs: 10, staleWhileRevalidate: false })
    cache.set('key1', 'value1')
    await new Promise(r => setTimeout(r, 15))
    expect(cache.get('key1')).toBeUndefined()
  })

  it('serves stale data within stale-while-revalidate window', async () => {
    const cache = new HttpCache({ ttlMs: 10, staleWhileRevalidate: true })
    cache.set('key1', 'value1')
    await new Promise(r => setTimeout(r, 15))
    const hit = cache.get('key1')
    expect(hit).toBeDefined()
    expect(hit!.data).toBe('value1')
    expect(hit!.stale).toBe(true)
    expect(cache.stats.staleHits).toBe(1)
  })

  it('invalidate removes specific key', () => {
    const cache = new HttpCache({ ttlMs: 1000 })
    cache.set('a', 1)
    cache.set('b', 2)
    cache.invalidate('a')
    expect(cache.get('a')).toBeUndefined()
    expect(cache.get('b')!.data).toBe(2)
  })

  it('invalidatePattern removes matching keys', () => {
    const cache = new HttpCache({ ttlMs: 1000 })
    cache.set('GET:/users', 1)
    cache.set('GET:/posts', 2)
    cache.set('POST:/items', 3)
    cache.invalidatePattern('^GET:')
    expect(cache.get('GET:/users')).toBeUndefined()
    expect(cache.get('GET:/posts')).toBeUndefined()
    expect(cache.get('POST:/items')!.data).toBe(3)
  })

  it('clear resets everything', () => {
    const cache = new HttpCache({ ttlMs: 1000 })
    cache.set('a', 1)
    cache.get('a')
    cache.clear()
    expect(cache.size).toBe(0)
    expect(cache.stats).toEqual({ hits: 0, misses: 0, staleHits: 0 })
  })

  it('makeKey generates deterministic key', () => {
    const k1 = HttpCache.makeKey('GET', '/users', { page: '1' })
    const k2 = HttpCache.makeKey('GET', '/users', { page: '1' })
    expect(k1).toBe(k2)
    expect(k1).toBe('GET:/users?page=1')
  })
})

describe('CircuitBreaker', () => {
  it('starts in closed state', () => {
    const cb = new CircuitBreaker({ failureThreshold: 3 })
    expect(cb.state).toBe('closed')
    expect(cb.allow()).toBe(true)
  })

  it('opens after threshold failures', () => {
    const cb = new CircuitBreaker({ failureThreshold: 3 })
    cb.recordFailure()
    cb.recordFailure()
    expect(cb.state).toBe('closed')
    cb.recordFailure()
    expect(cb.state).toBe('open')
    expect(cb.allow()).toBe(false)
  })

  it('transitions to half-open after reset timeout', async () => {
    const cb = new CircuitBreaker({ failureThreshold: 2, resetTimeoutMs: 20 })
    cb.recordFailure()
    cb.recordFailure()
    expect(cb.state).toBe('open')
    await new Promise(r => setTimeout(r, 25))
    expect(cb.state).toBe('half-open')
    expect(cb.allow()).toBe(true)
  })

  it('closes from half-open on success', async () => {
    const cb = new CircuitBreaker({ failureThreshold: 2, resetTimeoutMs: 10 })
    cb.recordFailure()
    cb.recordFailure()
    await new Promise(r => setTimeout(r, 30))
    cb.recordSuccess()
    expect(cb.state).toBe('closed')
  })

  it('does not reset failure count on success in closed state', () => {
    const cb = new CircuitBreaker({ failureThreshold: 5 })
    cb.recordFailure()
    cb.recordFailure()
    cb.recordSuccess()
    expect(cb.failureCount).toBe(2)
  })

  it('reset() returns to closed', () => {
    const cb = new CircuitBreaker({ failureThreshold: 2 })
    cb.recordFailure()
    cb.recordFailure()
    cb.reset()
    expect(cb.state).toBe('closed')
    expect(cb.failureCount).toBe(0)
  })

  it('half-open limits concurrent attempts', async () => {
    const cb = new CircuitBreaker({ failureThreshold: 1, resetTimeoutMs: 10, halfOpenMax: 1 })
    cb.recordFailure()
    await new Promise(r => setTimeout(r, 15))
    expect(cb.allow()).toBe(true) // first half-open attempt
    expect(cb.allow()).toBe(false) // second blocked
  })
})

describe('Throttler', () => {
  it('acquires immediately when under limit', async () => {
    const t = new Throttler({ maxConcurrent: 2 })
    await t.acquire()
    expect(t.running).toBe(1)
    await t.acquire()
    expect(t.running).toBe(2)
    t.release()
    expect(t.running).toBe(1)
    t.release()
    expect(t.running).toBe(0)
  })

  it('queues when at max concurrency', async () => {
    const t = new Throttler({ maxConcurrent: 1, queueTimeoutMs: 500 })
    await t.acquire()
    let acquired = false
    const p = t.acquire().then(() => { acquired = true })
    await new Promise(r => setTimeout(r, 10))
    expect(acquired).toBe(false)
    expect(t.queued).toBe(1)
    t.release()
    await p
    expect(acquired).toBe(true)
    expect(t.running).toBe(1)
    t.release()
  })

  it('rejects when queue is full', async () => {
    const t = new Throttler({ maxConcurrent: 1, maxQueue: 1, queueTimeoutMs: 100 })
    await t.acquire()
    t.acquire() // fills queue slot (will be queued, but we need to fill it first)
    // Wait for the first queued item to be added
    await new Promise(r => setTimeout(r, 5))
    await expect(t.acquire()).rejects.toThrow(ApiError)
    t.release()
  })
})

describe('httpClient singleton', () => {
  beforeEach(() => { vi.clearAllMocks(); mockFetch.mockReset() })

  it('has interceptors, cache, circuitBreaker, throttler', () => {
    expect(httpClient.interceptors).toBeDefined()
    expect(httpClient.cache).toBeDefined()
    expect(httpClient.circuitBreaker).toBeDefined()
    expect(httpClient.throttler).toBeDefined()
  })

  it('has hooks arrays', () => {
    expect(Array.isArray(httpClient.hooks.beforeRequest)).toBe(true)
    expect(Array.isArray(httpClient.hooks.afterResponse)).toBe(true)
    expect(Array.isArray(httpClient.hooks.onError)).toBe(true)
  })

  it('configure updates defaults', () => {
    httpClient.configure({ timeout: 10000 })
    expect(httpClient.defaults.timeout).toBe(10000)
    httpClient.configure({ timeout: 30000 }) // restore
  })

  it('get/post/put/patch/delete methods work', async () => {
    mockFetch.mockResolvedValue(mockOk({ ok: true }))
    const r1 = await httpClient.get('/test')
    expect(r1).toEqual({ ok: true })

    mockFetch.mockResolvedValue(mockOk({ created: true }))
    const r2 = await httpClient.post('/test', { x: 1 })
    expect(r2).toEqual({ created: true })

    mockFetch.mockResolvedValue(mockOk({ updated: true }))
    const r3 = await httpClient.put('/test', { x: 2 })
    expect(r3).toEqual({ updated: true })

    mockFetch.mockResolvedValue(mockOk({ patched: true }))
    const r4 = await httpClient.patch('/test', { x: 3 })
    expect(r4).toEqual({ patched: true })

    mockFetch.mockResolvedValue(mockOk({ deleted: true }))
    const r5 = await httpClient.delete('/test')
    expect(r5).toEqual({ deleted: true })
  })
})

describe('createHttpClient', () => {
  beforeEach(() => { vi.clearAllMocks(); mockFetch.mockReset() })

  it('creates isolated instance', () => {
    const c1 = createHttpClient()
    const c2 = createHttpClient()
    expect(c1).not.toBe(c2)
    expect(c1.interceptors).not.toBe(c2.interceptors)
  })

  it('accepts custom interceptors', async () => {
    let interceptorCalled = false
    const client = createHttpClient({
      interceptors: {
        request: [{
          onFulfilled: async (config) => { interceptorCalled = true; return config },
        }],
      },
    })
    mockFetch.mockResolvedValue(mockOk())
    await client.get('/test')
    expect(interceptorCalled).toBe(true)
  })

  it('accepts custom cache config', () => {
    const client = createHttpClient({ cache: { ttlMs: 5000 } })
    expect(client.cache).toBeDefined()
  })

  it('accepts custom circuit breaker config', () => {
    const client = createHttpClient({ circuitBreaker: { failureThreshold: 10 } })
    expect(client.circuitBreaker).toBeDefined()
    expect(client.circuitBreaker.failureCount).toBe(0)
  })
})

describe('apiGet with cache', () => {
  beforeEach(() => { vi.clearAllMocks(); mockFetch.mockReset() })

  it('caches GET response and returns from cache', async () => {
    mockFetch.mockResolvedValue(mockOk({ cached: true }))
    const r1 = await apiGet('/cached', undefined, { cache: { ttlMs: 5000 } })
    expect(r1).toEqual({ cached: true })
    expect(mockFetch).toHaveBeenCalledTimes(1)

    const r2 = await apiGet('/cached', undefined, { cache: { ttlMs: 5000 } })
    expect(r2).toEqual({ cached: true })
    expect(mockFetch).toHaveBeenCalledTimes(1) // still 1, cache hit
  })

  it('cache: true uses default TTL', async () => {
    mockFetch.mockResolvedValue(mockOk({ data: 1 }))
    await apiGet('/default-cache', undefined, { cache: true })
    expect(mockFetch).toHaveBeenCalledTimes(1)
    await apiGet('/default-cache', undefined, { cache: true })
    expect(mockFetch).toHaveBeenCalledTimes(1)
  })
})

describe('apiGet with dedupTtlMs', () => {
  beforeEach(() => { vi.clearAllMocks(); mockFetch.mockReset() })

  it('deduplicates identical GETs within TTL window', async () => {
    mockFetch.mockResolvedValue(mockOk({ deduped: true }))
    const p1 = apiGet('/dedup-test', undefined, { dedupTtlMs: 5000 })
    const p2 = apiGet('/dedup-test', undefined, { dedupTtlMs: 5000 })
    const [r1, r2] = await Promise.all([p1, p2])
    expect(mockFetch).toHaveBeenCalledTimes(1)
    expect(r1).toEqual({ deduped: true })
    expect(r2).toEqual({ deduped: true })
  })
})

describe('circuit breaker integration', () => {
  beforeEach(() => { vi.clearAllMocks(); mockFetch.mockReset() })

  it('records failures on 500 errors', async () => {
    const cb = httpClient.circuitBreaker
    cb.reset()
    mockFetch.mockResolvedValue(mockError(500))
    await expect(httpClient.get('/cb-test-1')).rejects.toThrow(ApiError)
    expect(cb.failureCount).toBeGreaterThanOrEqual(1)
    cb.reset()
  })
})

describe('httpClient with hooks', () => {
  beforeEach(() => { vi.clearAllMocks(); mockFetch.mockReset() })

  it('beforeRequest hook modifies config', async () => {
    const client = createHttpClient()
    let hookUrl = ''
    client.hooks.beforeRequest.push(async (config) => {
      hookUrl = config.url
      return config
    })
    mockFetch.mockResolvedValue(mockOk())
    await client.get('/hook-test')
    expect(hookUrl).toBe('/hook-test')
    client.hooks.beforeRequest.pop()
  })
})
