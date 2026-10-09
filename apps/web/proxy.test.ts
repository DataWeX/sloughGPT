import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { NextRequest } from 'next/server'

function makeRequest(pathname: string, method = 'GET'): NextRequest {
  return {
    nextUrl: { pathname },
    method,
    url: `http://localhost:3000${pathname}`,
    headers: new Headers(),
  } as unknown as NextRequest
}

describe('proxy', () => {
  beforeEach(() => {
    vi.unstubAllGlobals()
    vi.stubEnv('NODE_ENV', 'development')
  })

  it('passes through ignored paths', async () => {
    const { proxy } = await import('./proxy')
    const response = proxy(makeRequest('/_next/static/chunk.js'))
    expect(response.status).toBe(200)
  })

  it('sets X-Request-ID header', async () => {
    const { proxy } = await import('./proxy')
    const response = proxy(makeRequest('/chat', 'GET'))
    expect(response.headers.get('X-Request-ID')).toBeTruthy()
  })

  it('sets X-Response-Time header', async () => {
    const { proxy } = await import('./proxy')
    const response = proxy(makeRequest('/chat', 'GET'))
    expect(response.headers.get('X-Response-Time')).toMatch(/\d+ms/)
  })

  it('redirects known legacy paths', async () => {
    const { proxy } = await import('./proxy')
    vi.stubGlobal(
      'URL',
      class URL {
        constructor(public href: string) {}
      } as never,
    )

    const response = proxy(makeRequest('/infer', 'GET'))
    expect(response.status).toBeGreaterThanOrEqual(300)
    expect(response.status).toBeLessThan(400)
  })

  it('redirects every shared-table entry (no local fork of redirects.ts)', async () => {
    const { proxy } = await import('./proxy')
    const { REDIRECTS } = await import('./lib/redirects')
    vi.stubGlobal(
      'URL',
      class URL {
        constructor(public href: string) {}
      } as never,
    )

    const paths = Object.keys(REDIRECTS)
    expect(paths.length).toBeGreaterThanOrEqual(47) // count pin: keep in step
    for (const from of paths) {
      const response = proxy(makeRequest(from, 'GET'))
      expect(response.status, `${from} must redirect`).toBeGreaterThanOrEqual(300)
      expect(response.status, `${from} must redirect`).toBeLessThan(400)
    }
  })

  it('passes through unknown paths', async () => {
    const { proxy } = await import('./proxy')
    const response = proxy(makeRequest('/chat', 'POST'))
    expect(response.status).toBe(200)
  })
})
