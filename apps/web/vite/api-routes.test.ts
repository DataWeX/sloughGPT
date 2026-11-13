import { describe, expect, it } from 'vitest'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { discoverApiRoutes, isScopedApiPath, matchApiRoute, toPattern } from './api-routes'
import { invokeRouteHandler, toSsrUrl } from './api-middleware-plugin'

const webRoot = join(import.meta.dirname, '..')

describe('toPattern', () => {
  it('maps Next dynamic segments', () => {
    expect(toPattern('/api/planner/notes')).toBe('/api/planner/notes')
    expect(toPattern('/api/planner/notes/[id]')).toBe('/api/planner/notes/:id')
    expect(toPattern('/api/planner/board/cards/[id]')).toBe('/api/planner/board/cards/:id')
  })
})

describe('isScopedApiPath', () => {
  it('accepts planner and calendar only', () => {
    expect(isScopedApiPath('/api/planner/board')).toBe(true)
    expect(isScopedApiPath('/api/calendar/events')).toBe(true)
    expect(isScopedApiPath('/api/auth/providers')).toBe(false)
    expect(isScopedApiPath('/api/other')).toBe(false)
    expect(isScopedApiPath('/settings')).toBe(false)
  })
})

describe('discoverApiRoutes', () => {
  it('finds planner and calendar route.ts files with methods', () => {
    const routes = discoverApiRoutes(webRoot)
    expect(routes.length).toBeGreaterThanOrEqual(8)

    const board = routes.find((r) => r.pattern === '/api/planner/board')
    expect(board).toBeTruthy()
    expect(board!.methods).toContain('GET')

    const cards = routes.find((r) => r.pattern === '/api/planner/board/cards')
    expect(cards).toBeTruthy()
    expect(cards!.methods).toContain('POST')

    const cardId = routes.find((r) => r.pattern === '/api/planner/board/cards/:id')
    expect(cardId).toBeTruthy()
    expect(cardId!.methods).toEqual(expect.arrayContaining(['PUT', 'DELETE']))

    const events = routes.find((r) => r.pattern === '/api/calendar/events')
    expect(events).toBeTruthy()
    expect(events!.methods).toEqual(expect.arrayContaining(['GET', 'POST']))

    const notesId = routes.find((r) => r.pattern === '/api/planner/notes/:id')
    expect(notesId).toBeTruthy()

    for (const r of routes) {
      expect(existsSync(r.file)).toBe(true)
    }

    expect(routes.some((r) => r.pattern.includes('auth'))).toBe(false)
  })
})

describe('matchApiRoute', () => {
  const routes = discoverApiRoutes(webRoot)

  it('matches static and dynamic paths', () => {
    const board = matchApiRoute('/api/planner/board', routes)
    expect(board?.route.pattern).toBe('/api/planner/board')
    expect(board?.params).toEqual({})

    const note = matchApiRoute('/api/planner/notes/20260923_120000_hello', routes)
    expect(note?.route.pattern).toBe('/api/planner/notes/:id')
    expect(note?.params).toEqual({ id: '20260923_120000_hello' })

    const card = matchApiRoute('/api/planner/board/cards/card-1', routes)
    expect(card?.params).toEqual({ id: 'card-1' })

    expect(matchApiRoute('/api/planner/missing', routes)).toBeNull()
    expect(matchApiRoute('/api/auth/x', routes)).toBeNull()
  })
})

describe('toSsrUrl', () => {
  it('maps absolute file to root-relative ssr URL', () => {
    expect(
      toSsrUrl(
        '/home/x/sloughGPT/apps/web/app/api/planner/board/route.ts',
        '/home/x/sloughGPT/apps/web',
      ),
    ).toBe('/app/api/planner/board/route.ts')
  })
})

describe('invokeRouteHandler', () => {
  it('returns 405 when method not exported', async () => {
    const res = await invokeRouteHandler(
      { GET: () => Response.json({ ok: 1 }) },
      'POST',
      new Request('http://x'),
      {},
    )
    expect(res.status).toBe(405)
  })

  it('passes params promise to the handler', async () => {
    let seen: string | undefined
    const mod = {
      PUT: async (_req: Request, ctx?: { params?: Promise<Record<string, string>> }) => {
        const p = await ctx?.params
        seen = p?.id
        return Response.json({ id: p?.id })
      },
    }
    const res = await invokeRouteHandler(mod, 'PUT', new Request('http://x'), {
      id: 'abc',
    })
    expect(res.status).toBe(200)
    expect(seen).toBe('abc')
    expect(await res.json()).toEqual({ id: 'abc' })
  })
})
