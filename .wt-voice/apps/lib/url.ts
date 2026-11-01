import { PUBLIC_API_URL } from '../web/lib/config'

export function _isDocstoreUrl(url: string): boolean {
  return url.includes('/docstore/')
}

// Vite's apiRoutesPlugin serves the planner/calendar route handlers
// (apps/web/app/api/**/route.ts) on the WEB origin in dev. Prefixing these
// with PUBLIC_API_URL sends them to FastAPI, which has no such routes and
// answers 404 — keep them same-origin. Must match API_SCOPES in
// apps/web/vite/api-routes.ts (isScopedApiPath).
const SAME_ORIGIN_RE = /^\/api\/(planner|calendar)(\/|$)/

/** Resolve a request path against PUBLIC_API_URL, preserving scoped same-origin API routes. */
export function _resolveUrl(url: string): string {
  if (/^https?:\/\//.test(url)) return url // already absolute
  if (SAME_ORIGIN_RE.test(url)) return url // served by the web origin
  return `${PUBLIC_API_URL}${url}`
}
