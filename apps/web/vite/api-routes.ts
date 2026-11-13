// Discover and match Next-style app/api/**/route.ts files under Vite (dev).
// Scope: planner + calendar only (auth is Phase 4+).

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

export interface ApiRouteDef {
  /** Absolute path to route.ts */
  file: string
  /** URL path with :param / * segments, e.g. /api/planner/notes/:id */
  pattern: string
  /** HTTP methods exported by the module */
  methods: string[]
}

export interface ApiRouteMatch {
  route: ApiRouteDef
  params: Record<string, string>
}

const API_SCOPES = ['planner', 'calendar'] as const
const METHOD_RE = /export\s+(?:async\s+)?function\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b/g

export function discoverApiRoutes(webRoot: string): ApiRouteDef[] {
  const routes: ApiRouteDef[] = []
  for (const scope of API_SCOPES) {
    const root = join(webRoot, 'app', 'api', scope)
    if (!existsSync(root)) continue
    walk(root, routes, webRoot)
  }
  routes.sort((a, b) => b.pattern.length - a.pattern.length)
  return routes
}

function walk(dir: string, out: ApiRouteDef[], webRoot: string): void {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) {
      walk(full, out, webRoot)
      continue
    }
    if (entry !== 'route.ts' && entry !== 'route.tsx') continue
    const rel = relative(webRoot, full).replace(/\\/g, '/')
    let urlPath = rel.replace(/^app\//, '/').replace(/\/route\.tsx?$/, '')
    if (urlPath.endsWith('/')) urlPath = urlPath.slice(0, -1)
    const pattern = toPattern(urlPath)
    if (!pattern.startsWith('/api/planner') && !pattern.startsWith('/api/calendar')) continue
    if (pattern.includes('...')) continue
    out.push({
      file: full,
      pattern,
      methods: parseMethods(full),
    })
  }
}

/** `/api/planner/notes/[id]` -> `/api/planner/notes/:id` */
export function toPattern(urlPath: string): string {
  return urlPath
    .split('/')
    .map((seg) => {
      if (!seg.startsWith('[') || !seg.endsWith(']')) return seg
      const inner = seg.slice(1, -1)
      if (inner.startsWith('...')) return `*${inner.slice(3)}`
      return `:${inner}`
    })
    .join('/')
}

function parseMethods(file: string): string[] {
  const src = readFileSync(file, 'utf-8')
  const methods = new Set<string>()
  let m: RegExpExecArray | null
  METHOD_RE.lastIndex = 0
  while ((m = METHOD_RE.exec(src))) methods.add(m[1])
  return [...methods]
}

export function matchApiRoute(pathname: string, routes: ApiRouteDef[]): ApiRouteMatch | null {
  const parts = splitPath(pathname)
  for (const route of routes) {
    const patternParts = splitPath(route.pattern)
    const params = matchParts(parts, patternParts)
    if (params) return { route, params }
  }
  return null
}

function splitPath(p: string): string[] {
  return p.split('/').filter(Boolean)
}

function matchParts(actual: string[], pattern: string[]): Record<string, string> | null {
  if (actual.length !== pattern.length) return null
  const params: Record<string, string> = {}
  for (let i = 0; i < pattern.length; i++) {
    const p = pattern[i]
    const a = actual[i]
    if (p.startsWith(':')) {
      params[p.slice(1)] = decodeURIComponent(a)
      continue
    }
    if (p !== a) return null
  }
  return params
}

export function isScopedApiPath(pathname: string): boolean {
  return /^\/api\/(planner|calendar)(\/|$)/.test(pathname)
}
