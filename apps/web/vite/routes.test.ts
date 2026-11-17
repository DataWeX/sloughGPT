import { describe, expect, it } from 'vitest'
import { globToRoutePath, discoveredRoutes, routePathList } from './routes'

describe('globToRoutePath', () => {
  it('maps App Router paths', () => {
    expect(globToRoutePath('../app/(app)/page.tsx')).toBe('/')
    expect(globToRoutePath('../app/(app)/settings/page.tsx')).toBe('/settings')
    // /knowledge/[id] retired by route-map card 7/7 (folded into /knowledge?id=);
    // training/job/[id] below still covers the dynamic [id] -> :id conversion.
    expect(globToRoutePath('../app/(app)/consciousness/settings/page.tsx')).toBe(
      '/consciousness/settings',
    )
    expect(globToRoutePath('../app/(app)/training/job/[id]/page.tsx')).toBe('/training/job/:id')
  })
})

describe('discoveredRoutes', () => {
  it('finds real pages including /monitoring', () => {
    expect(routePathList.length).toBeGreaterThan(50)
    expect(routePathList).toContain('/monitoring')
    expect(routePathList).toContain('/')
    expect(routePathList).toContain('/models')
    expect(discoveredRoutes.every((r) => r.Component)).toBe(true)
  })
})

describe('companion resolution (loading/error wiring)', () => {
  const byPath = new Map(discoveredRoutes.map((r) => [r.path, r]))

  it('resolves a loading fallback and an error component for EVERY route', () => {
    // Every page's directory chain reaches app/(app)/ or app/, which carry
    // both shared companions — no route may fall through to the generic
    // PageFallback or to an unwrapped error.
    expect(discoveredRoutes.every((r) => r.Loading)).toBe(true)
    expect(discoveredRoutes.every((r) => r.ErrorComponent)).toBe(true)
  })

  it('prefers route-local companions over shared ones', () => {
    const chat = byPath.get('/chat') // local loading.tsx + error.tsx
    const job = byPath.get('/training/job/:id') // local pair
    const kanban = byPath.get('/kanban') // no local companions at all
    expect(chat?.Loading).toBeTruthy()
    expect(job?.Loading).toBeTruthy()
    expect(kanban?.Loading).toBeTruthy()
    // Local pairs are distinct components from the inherited (app) pair.
    expect(job?.Loading).not.toBe(kanban?.Loading)
    expect(chat?.ErrorComponent).not.toBe(kanban?.ErrorComponent)
  })

  it('inherits the nearest ancestor when a route has no local companion', () => {
    const profile = byPath.get('/profile') // local error.tsx, NO loading.tsx
    const kanban = byPath.get('/kanban') // neither
    const workspaceSettings = byPath.get('/workspace/settings') // neither
    expect(profile).toBeTruthy()
    expect(workspaceSettings).toBeTruthy()
    // loading: profile and kanban both inherit app/(app)/loading.tsx;
    // chat's is local — a different component.
    expect(profile?.Loading).toBe(kanban?.Loading)
    expect(kanban?.Loading).not.toBe(byPath.get('/chat')?.Loading)
    // error: kanban and workspace/settings both inherit app/(app)/error.tsx;
    // profile's is its own app/(app)/profile/error.tsx — a different component.
    expect(kanban?.ErrorComponent).toBe(workspaceSettings?.ErrorComponent)
    expect(profile?.ErrorComponent).not.toBe(kanban?.ErrorComponent)
  })
})
