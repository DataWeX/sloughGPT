import { PUBLIC_API_URL } from '../web/lib/config'

export function _isDocstoreUrl(url: string): boolean {
  return url.includes('/docstore/')
}

// The planner/calendar handlers live in the Python backend now
// (apps/api/server/routers/planner.py + calendar.py). Same-origin in dev via
// vite's server.proxy (vite.config.ts), same-origin in prod via the gateway —
// prefixing these with PUBLIC_API_URL would only matter for absolute bases, so
// keep them relative. Must match /api/(planner|calendar) below.
const SAME_ORIGIN_RE = /^\/api\/(planner|calendar)(\/|$)/

/** Resolve a request path against PUBLIC_API_URL, preserving scoped same-origin API routes. */
export function _resolveUrl(url: string): string {
  if (/^https?:\/\//.test(url)) return url // already absolute
  if (SAME_ORIGIN_RE.test(url)) return url // served by the web origin
  return `${PUBLIC_API_URL}${url}`
}
