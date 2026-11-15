import { describe, it, expect, vi, beforeEach } from 'vitest'
import { proxyRequest } from './planner-proxy'

vi.mock('@/vite/next-compat/server', () => {
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

/** The first-party relay marker every call site must set (card 20261108_142919). */
const RELAY = { 'x-slo-relay': '1' }

function fakeResponse(status: number, text: string, headers: Record<string, string> = {}) {
  return {
    ok: status < 400,
    status,
    text: () => Promise.resolve(text),
    headers: {
      get: (name: string) =>
        name.toLowerCase() === 'content-type' ? (headers[name] ?? null) : null,
    },
  }
}

beforeEach(() => {
  mockFetch.mockReset()
})

describe('proxyRequest', () => {
  it('proxies GET request to planner backend', async () => {
    mockFetch.mockResolvedValue(
      fakeResponse(200, '{"data":"test"}', { 'content-type': 'application/json' }),
    )

    const res = await proxyRequest('/board', { headers: RELAY })
    expect(mockFetch).toHaveBeenCalledWith('http://127.0.0.1:8787/board', {
      method: 'GET',
      body: undefined,
      // the relay header is consumed by the proxy, never forwarded downstream
      headers: undefined,
    })
    expect(res.status).toBe(200)
    expect(res.body).toBe('{"data":"test"}')
  })

  it('proxies POST request with body and custom headers', async () => {
    mockFetch.mockResolvedValue(
      fakeResponse(201, '{"id":"new"}', { 'content-type': 'application/json' }),
    )

    const res = await proxyRequest('/cards', {
      method: 'POST',
      body: JSON.stringify({ title: 'Test' }),
      headers: { 'Content-Type': 'application/json', ...RELAY },
    })
    expect(mockFetch).toHaveBeenCalledWith('http://127.0.0.1:8787/cards', {
      method: 'POST',
      body: JSON.stringify({ title: 'Test' }),
      // relay header stripped; app headers pass through untouched
      headers: { 'Content-Type': 'application/json' },
    })
    expect(res.status).toBe(201)
    expect(res.body).toBe('{"id":"new"}')
  })

  it('returns 502 when backend is unreachable', async () => {
    mockFetch.mockRejectedValue(new Error('Connection refused'))

    const res = await proxyRequest('/board', { headers: RELAY })
    expect(res.status).toBe(502)
    expect(res.body).toContain('Planner backend unreachable')
  })

  it('preserves backend error status codes', async () => {
    mockFetch.mockResolvedValue(
      fakeResponse(404, '{"error":"Not found"}', { 'content-type': 'application/json' }),
    )

    const res = await proxyRequest('/cards/nonexistent', { headers: RELAY })
    expect(res.status).toBe(404)
    expect(res.body).toBe('{"error":"Not found"}')
  })

  it('defaults non-JSON responses to text/plain', async () => {
    mockFetch.mockResolvedValue(fakeResponse(200, 'plain text', {}))

    const res = await proxyRequest('/health', { headers: RELAY })
    expect(res.status).toBe(200)
    expect((res.headers as unknown as Record<string, string>)['content-type']).toBe('text/plain')
  })

  it('rejects a relay without the first-party header (403, no fetch)', async () => {
    const res = await proxyRequest('/board')
    expect(res.status).toBe(403)
    expect(res.body).toContain('X-Slo-Relay')
    expect(mockFetch).not.toHaveBeenCalled()
  })

  it('rejects an empty/absent relay value (403, no fetch)', async () => {
    const res = await proxyRequest('/board', { headers: { 'x-slo-relay': '' } })
    expect(res.status).toBe(403)
    expect(mockFetch).not.toHaveBeenCalled()
  })

  it('rejects a path that escapes the pinned planner host (403, no fetch)', async () => {
    // `http://127.0.0.1:8787@evil.com/...` parses with host evil.com — the
    // userinfo@ trick must be caught by the host pin, not forwarded.
    const res = await proxyRequest('@evil.com/steal', { headers: RELAY })
    expect(res.status).toBe(403)
    expect(res.body).toContain('host pin')
    expect(mockFetch).not.toHaveBeenCalled()
  })

  it('keeps a protocol-relative path on the pinned host', async () => {
    // `http://127.0.0.1:8787//evil.com/...` parses with host 127.0.0.1:8787
    // (authority ends at the first slash) — the fetch must stay pinned there.
    mockFetch.mockResolvedValue(fakeResponse(200, '{"ok":true}', {}))

    const res = await proxyRequest('//evil.com/steal', { headers: RELAY })
    expect(res.status).toBe(200)
    const target = mockFetch.mock.calls[0]?.[0] as string
    expect(new URL(target).host).toBe('127.0.0.1:8787')
  })
})
