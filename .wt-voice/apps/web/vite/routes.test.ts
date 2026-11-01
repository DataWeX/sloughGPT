import { describe, expect, it } from 'vitest'
import { globToRoutePath, discoveredRoutes, routePathList } from './routes'

describe('globToRoutePath', () => {
  it('maps App Router paths', () => {
    expect(globToRoutePath('../app/(app)/page.tsx')).toBe('/')
    expect(globToRoutePath('../app/(app)/settings/page.tsx')).toBe('/settings')
    expect(globToRoutePath('../app/(app)/model/[id]/page.tsx')).toBe('/model/:id')
    expect(globToRoutePath('../app/(app)/consciousness/settings/page.tsx')).toBe(
      '/consciousness/settings',
    )
    expect(globToRoutePath('../app/(app)/training/job/[id]/page.tsx')).toBe('/training/job/:id')
  })
})

describe('discoveredRoutes', () => {
  it('finds real pages including /errors and /monitoring', () => {
    expect(routePathList.length).toBeGreaterThan(50)
    expect(routePathList).toContain('/errors')
    expect(routePathList).toContain('/monitoring')
    expect(routePathList).toContain('/')
    expect(routePathList).toContain('/model/:id')
    expect(discoveredRoutes.every((r) => r.Component)).toBe(true)
  })
})
