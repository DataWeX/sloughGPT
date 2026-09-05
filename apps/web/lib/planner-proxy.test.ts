import { describe, it, expect, vi, beforeEach } from 'vitest'
import { proxyRequest } from './planner-proxy'

vi.mock('next/server', () => {
  class NextResponse {
    body: unknown
    status: number
    headers: Record<string, string>
    ok: boolean
    constructor(body: unknown, init: { status?: number; headers?: Record<string, string> } = {}) {
      this.body = body
      this.status = init.status ?? 200
      this.headers = init.headers ?? {}
      this.ok = (init.status ?? 200) < 400
    }
    static json(body: unknown, init?: { status?: number }) {
      return new NextResponse(JSON.stringify(body), init)
    }
  }
  return { NextResponse }
})

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

function fakeResponse(status: number, text: string, headers: Record<string, string> = {}) {
  return {
    ok: status < 400,
    status,
    text: () => Promise.resolve(text),
    headers: { get: (name: string) => (name.toLowerCase() === 'content-type' ? (headers[name] ?? null) : null) },
  }
}

beforeEach(() => {
  mockFetch.mockReset()
})

describe('proxyRequest', () => {
  it('proxies GET request to planner backend', async () => {
    mockFetch.mockResolvedValue(fakeResponse(200, '{"data":"test"}', { 'content-type': 'application/json' }))

    const res = await proxyRequest('/board')
    expect(mockFetch).toHaveBeenCalledWith('http://127.0.0.1:8787/board', {
      method: 'GET',
      body: undefined,
      headers: undefined,
    })
    expect(res.status).toBe(200)
    expect(res.body).toBe('{"data":"test"}')
  })

  it('proxies POST request with body and custom headers', async () => {
    mockFetch.mockResolvedValue(fakeResponse(201, '{"id":"new"}', { 'content-type': 'application/json' }))

    const res = await proxyRequest('/cards', {
      method: 'POST',
      body: JSON.stringify({ title: 'Test' }),
      headers: { 'Content-Type': 'application/json' },
    })
    expect(mockFetch).toHaveBeenCalledWith('http://127.0.0.1:8787/cards', {
      method: 'POST',
      body: JSON.stringify({ title: 'Test' }),
      headers: { 'Content-Type': 'application/json' },
    })
    expect(res.status).toBe(201)
    expect(res.body).toBe('{"id":"new"}')
  })

  it('returns 502 when backend is unreachable', async () => {
    mockFetch.mockRejectedValue(new Error('Connection refused'))

    const res = await proxyRequest('/board')
    expect(res.status).toBe(502)
    expect(res.body).toContain('Planner backend unreachable')
  })

  it('preserves backend error status codes', async () => {
    mockFetch.mockResolvedValue(fakeResponse(404, '{"error":"Not found"}', { 'content-type': 'application/json' }))

    const res = await proxyRequest('/cards/nonexistent')
    expect(res.status).toBe(404)
    expect(res.body).toBe('{"error":"Not found"}')
  })

  it('defaults non-JSON responses to text/plain', async () => {
    mockFetch.mockResolvedValue(fakeResponse(200, 'plain text', {}))

    const res = await proxyRequest('/health')
    expect(res.status).toBe(200)
    expect(res.headers['content-type']).toBe('text/plain')
  })
})
