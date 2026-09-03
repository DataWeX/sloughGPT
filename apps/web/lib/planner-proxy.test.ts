import { describe, it, expect, vi, beforeEach } from 'vitest'
import { proxyRequest, PLANNER_URL } from './planner-proxy'

// Mock NextResponse
vi.mock('next/server', () => ({
  NextResponse: {
    json: (body: unknown, init?: { status?: number }) => ({
      body,
      status: init?.status ?? 200,
      ok: (init?.status ?? 200) < 400,
    }),
  },
}))

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

beforeEach(() => {
  mockFetch.mockReset()
})

describe('PLANNER_URL', () => {
  it('defaults to localhost:8787', () => {
    expect(PLANNER_URL).toBe('http://127.0.0.1:8787')
  })
})

describe('proxyRequest', () => {
  it('proxies GET request to planner backend', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ data: 'test' }),
    })

    const res = await proxyRequest('/board')
    expect(mockFetch).toHaveBeenCalledWith('http://127.0.0.1:8787/board', {
      headers: { 'Content-Type': 'application/json' },
    })
    expect(res.body).toEqual({ data: 'test' })
    expect(res.status).toBe(200)
  })

  it('proxies POST request with body', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 201,
      json: () => Promise.resolve({ id: 'new' }),
    })

    const res = await proxyRequest('/cards', {
      method: 'POST',
      body: JSON.stringify({ title: 'Test' }),
    })
    expect(mockFetch).toHaveBeenCalledWith('http://127.0.0.1:8787/cards', {
      method: 'POST',
      body: JSON.stringify({ title: 'Test' }),
      headers: { 'Content-Type': 'application/json' },
    })
    expect(res.body).toEqual({ id: 'new' })
    expect(res.status).toBe(201)
  })

  it('returns 503 when backend is unavailable', async () => {
    mockFetch.mockRejectedValue(new Error('Connection refused'))

    const res = await proxyRequest('/board')
    expect(res.status).toBe(503)
    expect((res.body as unknown as Record<string, string>).error).toContain('backend unavailable')
  })

  it('preserves backend error status codes', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 404,
      json: () => Promise.resolve({ error: 'Not found' }),
    })

    const res = await proxyRequest('/cards/nonexistent')
    expect(res.status).toBe(404)
    expect(res.body).toEqual({ error: 'Not found' })
  })

  it('merges custom headers', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({}),
    })

    await proxyRequest('/board', {
      headers: { 'X-Custom': 'value' },
    })
    expect(mockFetch).toHaveBeenCalledWith('http://127.0.0.1:8787/board', {
      headers: { 'Content-Type': 'application/json', 'X-Custom': 'value' },
    })
  })
})
