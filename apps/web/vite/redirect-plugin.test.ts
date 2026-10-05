import { describe, expect, it } from 'vitest'
import { resolveRedirect, splitTarget, isIgnoredPath, REDIRECTS } from '../lib/redirects'
import { applyRedirect } from './redirect-plugin'
import type { ServerResponse } from 'node:http'

function mockRes() {
  const headers: Record<string, string> = {}
  const res = {
    statusCode: 200,
    ended: false,
    setHeader(k: string, v: string) {
      headers[k.toLowerCase()] = v
    },
    end() {
      this.ended = true
    },
    _headers: headers,
  }
  return res as typeof res & ServerResponse
}

describe('resolveRedirect / splitTarget', () => {
  it('maps legacy paths', () => {
    expect(resolveRedirect('/collections')).toBe('/datasets')
    expect(resolveRedirect('/voice')).toBe('/chat?mode=talk')
    expect(resolveRedirect('/errors')).toBe('/monitoring')
    expect(resolveRedirect('/settings')).toBeNull()
    expect(resolveRedirect('/_next/static/x.js')).toBeNull()
    expect(isIgnoredPath('/_next/static/x.js')).toBe(true)
  })

  it('splits query targets', () => {
    expect(splitTarget('/chat?mode=talk')).toEqual({ path: '/chat', search: '?mode=talk' })
    expect(splitTarget('/datasets')).toEqual({ path: '/datasets', search: '' })
  })

  it('has 47 entries matching proxy table', () => {
    expect(Object.keys(REDIRECTS).length).toBe(47)
  })

  it('maps workspace legacy paths to consolidated routes', () => {
    expect(resolveRedirect('/workspace-dashboard')).toBe('/workspace')
    expect(resolveRedirect('/usage')).toBe('/workspace/usage')
    expect(resolveRedirect('/audit-trail')).toBe('/workspace/audit')
    expect(resolveRedirect('/members')).toBe('/workspace/members')
    expect(resolveRedirect('/permissions')).toBe('/workspace/members/permissions')
    expect(resolveRedirect('/workspace-settings')).toBe('/workspace/settings')
    expect(resolveRedirect('/api-keys')).toBe('/workspace/settings/api-keys')
    expect(resolveRedirect('/notifications')).toBe('/workspace/settings/notifications')
    expect(resolveRedirect('/shared-data')).toBe('/workspace/data')
    expect(resolveRedirect('/workspace-search')).toBe('/workspace/data/search')
  })
})

describe('applyRedirect', () => {
  it('writes 307 + location and returns true', () => {
    const res = mockRes()
    expect(applyRedirect('/infer', res)).toBe(true)
    expect(res.statusCode).toBe(307)
    expect(res._headers.location).toBe('/models')
    expect(res.ended).toBe(true)
  })

  it('is a no-op for normal and ignored paths', () => {
    const a = mockRes()
    expect(applyRedirect('/settings', a)).toBe(false)
    expect(a.ended).toBe(false)

    const b = mockRes()
    expect(applyRedirect('/_next/static/a.js', b)).toBe(false)
    expect(b.ended).toBe(false)

    const c = mockRes()
    expect(applyRedirect('/api/planner/board', c)).toBe(false)
  })
})
