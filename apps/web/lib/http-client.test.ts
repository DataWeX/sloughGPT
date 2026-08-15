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

import { ApiError, apiGet, apiPost, apiPut, apiDelete, apiPatch, createApiClient } from './http-client'

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
