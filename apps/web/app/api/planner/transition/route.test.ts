import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { NextRequest } from '@/vite/next-compat/server'
import { proxyRequest } from '@/lib/planner-proxy'
import { POST } from './route'

vi.mock('@/lib/planner-proxy', () => ({ proxyRequest: vi.fn() }))

function makeRequest(body: unknown): NextRequest {
  return new Request('http://test.local/api/planner/transition', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  }) as unknown as NextRequest
}

describe('POST /api/planner/transition', () => {
  beforeEach(() => vi.clearAllMocks())

  it('forwards the payload to the engine-backed board route via proxyRequest', async () => {
    const payload = { card_id: 'c1', to_state: 'wip', actor: 'test' }
    const expected = new Response(JSON.stringify({ ok: true, card: { id: 'c1' } }), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    })
    vi.mocked(proxyRequest).mockResolvedValue(expected)

    const res = await POST(makeRequest(payload))

    expect(proxyRequest).toHaveBeenCalledWith('/api/board/transition', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
    expect(res).toBe(expected)
  })

  it('passes the proxy response through untouched (409 wip_exceeded rides it)', async () => {
    const rejected = new Response(
      JSON.stringify({ ok: false, error: 'wip_exceeded', legal_targets: ['todo'] }),
      { status: 409, headers: { 'content-type': 'application/json' } },
    )
    vi.mocked(proxyRequest).mockResolvedValue(rejected)

    const res = await POST(makeRequest({ card_id: 'c1', to_state: 'wip' }))

    expect(res.status).toBe(409)
    await expect(res.json()).resolves.toMatchObject({
      ok: false,
      error: 'wip_exceeded',
      legal_targets: ['todo'],
    })
  })
})
