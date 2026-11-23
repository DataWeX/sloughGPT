// Auto-wire every App Router page under app/<...>/page.tsx into react-router paths.
// NOTE: Do NOT put (app) in the glob pattern — () is extglob and matches nothing.
// Route groups are stripped in globToRoutePath instead.
// 2026-11-17 (route-companions card): loading.tsx/error.tsx companions are wired
// too — nearest-ancestor resolution per the Next segment rule.
import { lazy, type ComponentType, type LazyExoticComponent } from 'react'

import type { RouteErrorProps } from './RouteErrorBoundary'

export type PageGlob = { default: ComponentType<Record<string, unknown>> }

/** Vite glob relative to apps/web (this module lives at apps/web/vite/routes.ts). */
const pageModules = import.meta.glob('../app/**/page.tsx') as Record<string, () => Promise<unknown>>

// Companions load EAGERLY: a suspended route's fallback must exist the instant
// its page chunk starts loading (a lazy fallback that suspends renders blank,
// not a skeleton), and an error UI must be present synchronously at catch time.
// 23 + 25 thin wrappers — cheap, and they share PageSkeleton/CustomErrorHandler.
const loadingModules = import.meta.glob('../app/**/loading.tsx', {
  eager: true,
}) as Record<string, { default: ComponentType<Record<string, unknown>> }>
const errorModules = import.meta.glob('../app/**/error.tsx', {
  eager: true,
}) as Record<string, { default: ComponentType<RouteErrorProps> }>

export function globToRoutePath(globPath: string): string {
  // '../app/(app)/settings/page.tsx' | '../app/(app)/page.tsx' | '../app/(app)/training/job/[id]/page.tsx'
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
  /** Nearest loading.tsx at or above the page's directory (Next segment rule). */
  Loading?: ComponentType<Record<string, unknown>>
  /** Nearest error.tsx at or above the page's directory — speaks {error, reset}. */
  ErrorComponent?: ComponentType<RouteErrorProps>
}

/** '../app/(app)/chat/loading.tsx' -> '../app/(app)/chat' */
function companionDir(globPath: string): string {
  return globPath.slice(0, globPath.lastIndexOf('/'))
}

function buildCompanionIndex<T>(
  modules: Record<string, { default: ComponentType<T> }>,
): Map<string, ComponentType<T>> {
  const index = new Map<string, ComponentType<T>>()
  for (const [globPath, mod] of Object.entries(modules)) {
    index.set(companionDir(globPath), mod.default)
  }
  return index
}

/** Walk the page's directory chain up to ../app — first companion wins. */
function resolveCompanion<T>(
  pageGlobPath: string,
  index: Map<string, ComponentType<T>>,
): ComponentType<T> | undefined {
  let dir = companionDir(pageGlobPath)
  for (;;) {
    const hit = index.get(dir)
    if (hit) return hit
    if (dir === '../app') return undefined
    const cut = dir.lastIndexOf('/')
    if (cut <= 0) return undefined
    dir = dir.slice(0, cut)
  }
}

function buildRoutes(): DiscoveredRoute[] {
  const loadingIndex = buildCompanionIndex(loadingModules)
  const errorIndex = buildCompanionIndex(errorModules)
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
    seen.set(path, {
      path,
      Component: lazy(load),
      Loading: resolveCompanion(globPath, loadingIndex),
      ErrorComponent: resolveCompanion(globPath, errorIndex),
    })
  }
  return [...seen.values()].sort((a, b) => a.path.localeCompare(b.path))
}

export const discoveredRoutes: DiscoveredRoute[] = buildRoutes()

export const routePathList: string[] = discoveredRoutes.map((r) => r.path)

if (import.meta.env.DEV && typeof window !== 'undefined') {
  ;(window as unknown as { __vite_routes?: string[] }).__vite_routes = routePathList
}
