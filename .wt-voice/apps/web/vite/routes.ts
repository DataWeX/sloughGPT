// Auto-wire every App Router page under app/<...>/page.tsx into react-router paths.
// NOTE: Do NOT put (app) in the glob pattern — () is extglob and matches nothing.
// Route groups are stripped in globToRoutePath instead.
import { lazy, type ComponentType, type LazyExoticComponent } from 'react'

export type PageGlob = { default: ComponentType<Record<string, unknown>> }

/** Vite glob relative to apps/web (this module lives at apps/web/vite/routes.ts). */
const pageModules = import.meta.glob('../app/**/page.tsx') as Record<string, () => Promise<unknown>>

export function globToRoutePath(globPath: string): string {
  // '../app/(app)/settings/page.tsx' | '../app/(app)/page.tsx' | '../app/(app)/model/[id]/page.tsx'
  let p = globPath.replace(/^\.\.\/app\//, '')
  // Strip App Router route groups: (app)/, (marketing)/, …
  p = p.replace(/\([^)]+\)\//g, '')
  p = p.replace(/\/page\.tsx$/, '')
  p = p.replace(/^page\.tsx$/, '')
  if (p === '' || p === '.') return '/'
  p = p.replace(/\[([^\]]+)\]/g, ':$1')
  return p.startsWith('/') ? p : `/${p}`
}

export type DiscoveredRoute = {
  path: string
  Component: LazyExoticComponent<ComponentType<Record<string, unknown>>>
}

function buildRoutes(): DiscoveredRoute[] {
  const seen = new Map<string, DiscoveredRoute>()
  for (const [globPath, rawLoad] of Object.entries(pageModules)) {
    const path = globToRoutePath(globPath)
    if (seen.has(path)) continue
    const load = async () => {
      const mod = (await rawLoad()) as PageGlob
      if (!mod || typeof mod.default !== 'function') {
        throw new Error(`Page module missing default export: ${globPath}`)
      }
      return { default: mod.default }
    }
    seen.set(path, { path, Component: lazy(load) })
  }
  return [...seen.values()].sort((a, b) => a.path.localeCompare(b.path))
}

export const discoveredRoutes: DiscoveredRoute[] = buildRoutes()

export const routePathList: string[] = discoveredRoutes.map((r) => r.path)

if (import.meta.env.DEV && typeof window !== 'undefined') {
  ;(window as unknown as { __vite_routes?: string[] }).__vite_routes = routePathList
}
