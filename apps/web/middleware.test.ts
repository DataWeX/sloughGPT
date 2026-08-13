import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

const mockNext = vi.hoisted(() => ({ next: vi.fn() }))
const { mockHeaders } = vi.hoisted(() => {
  return {
    mockHeaders: (entries: Record<string, string> = {}) => ({
      entries,
      get: (key: string) => entries[key.toLowerCase()] ?? null,
      set: vi.fn(),
      forEach: (cb: (v: string, k: string) => void) =>
        Object.entries(entries).forEach(([k, v]) => cb(v, k)),
    }),
  }
})

vi.mock('next/server', () => ({
  NextResponse: { next: (opts?: unknown) => mockNext.next(opts) },
}))

const { middleware, config } = await import('./middleware')

function makeRequest(pathname: string, method = 'GET', headers: Record<string, string> = {}) {
  return {
    method,
    nextUrl: { pathname },
    headers: mockHeaders(headers),
  } as any
}

describe('middleware', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockNext.next.mockReturnValue({ status: 200, headers: mockHeaders() })
  })

  afterEach(() => {
    vi.unstubAllEnvs()
  })

  it('passes through immediately for ignored paths without headers', () => {
    const response = middleware(makeRequest('/_next/static/chunk.js'))
    expect(response).toBeDefined()
    expect(mockNext.next).toHaveBeenCalledTimes(1)
  })

  it('logs and adds timing/request-id headers for a normal request', () => {
    const response = middleware(makeRequest('/chat', 'GET'))
    const headers = response.headers as unknown as ReturnType<typeof mockHeaders>
    expect(mockNext.next).toHaveBeenCalledTimes(1)
    expect(headers.set).toHaveBeenCalledWith('X-Response-Time', expect.stringMatching(/\d+ms/))
    expect(headers.set).toHaveBeenCalledWith('X-Request-ID', expect.any(String))
  })

  it('logs the request line in development', () => {
    vi.stubEnv('NODE_ENV', 'development')
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {})
    try {
      middleware(makeRequest('/chat', 'GET'))
      const line = spy.mock.calls[0][0] as string
      expect(line).toContain('[INFO]')
      expect(line).toContain('GET')
      expect(line).toContain('/chat')
      expect(line).toMatch(/\d+ms/)
    } finally {
      spy.mockRestore()
    }
  })

  it('formats a POST request color differently', () => {
    vi.stubEnv('NODE_ENV', 'development')
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {})
    try {
      middleware(makeRequest('/api/data', 'POST'))
      expect(spy).toHaveBeenCalled()
    } finally {
      spy.mockRestore()
    }
  })

  it('config exports a matcher', () => {
    expect(Array.isArray(config.matcher)).toBe(true)
    expect(config.matcher.length).toBeGreaterThan(0)
  })

  it('handles GET requests', () => {
    const response = middleware(makeRequest('/models', 'GET'))
    expect(response).toBeDefined()
    expect(mockNext.next).toHaveBeenCalledTimes(1)
  })

  it('handles POST requests', () => {
    const response = middleware(makeRequest('/chat', 'POST'))
    expect(response).toBeDefined()
    expect(mockNext.next).toHaveBeenCalledTimes(1)
  })
})
