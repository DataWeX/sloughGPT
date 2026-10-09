// Shared legacy-path redirects — single source of truth for proxy.ts (Next)
// and the Vite dev/static middleware (Phase 4).

export const IGNORE_PATHS = ['/_next', '/favicon', '/sw.js', '/workbox']

export const REDIRECTS: Record<string, string> = {
  '/companion': '/souls',
  '/evaluate': '/benchmark',
  '/compare': '/benchmark',
  '/lora-eval': '/benchmark',
  '/experiments': '/benchmark',
  '/token-tree': '/tokenizer',
  '/meta-weights': '/models',
  '/infer': '/models',
  '/world': '/models',
  '/registry': '/models',
  '/memory': '/knowledge',
  '/kb': '/knowledge',
  '/docstore': '/knowledge',
  '/collections': '/datasets',
  '/self-train': '/training',
  '/learn': '/training',
  '/auto-train': '/training',
  '/rate-limit': '/monitoring',
  '/admin': '/settings',
  '/auth': '/settings',
  '/export': '/settings',
  '/errors': '/monitoring',
  '/security': '/monitoring',
  '/images': '/developer',
  '/session': '/chat',
  '/files': '/developer',
  '/voice': '/chat?mode=talk',
  '/writing': '/chat?mode=write',
  '/rewrite': '/chat?mode=rewrite',
  '/translate': '/chat?mode=translate',
  '/brainstorm': '/chat?mode=brainstorm',
  '/decide': '/chat?mode=decide',
  '/explain': '/chat?mode=explain',
  '/wellness': '/chat?mode=wellness',
  '/shell': '/developer',
  '/workflow': '/feedback',
  '/tools': '/chat',
  '/workspace-dashboard': '/workspace',
  '/usage': '/workspace/usage',
  '/audit-trail': '/workspace/audit',
  '/members': '/workspace/members',
  '/permissions': '/workspace/members/permissions',
  '/workspace-settings': '/workspace/settings',
  '/api-keys': '/workspace/settings/api-keys',
  '/notifications': '/workspace/settings/notifications',
  '/shared-data': '/workspace/data',
  '/workspace-search': '/workspace/data/search',
}

export function isIgnoredPath(pathname: string): boolean {
  return IGNORE_PATHS.some((p) => pathname.startsWith(p))
}

/** Exact-path redirect lookup. Target may include `?query`. */
export function resolveRedirect(pathname: string): string | null {
  if (isIgnoredPath(pathname)) return null
  return REDIRECTS[pathname] ?? null
}

/** Split a redirect target into path + search (for react-router `<Navigate to>`). */
export function splitTarget(target: string): { path: string; search: string } {
  const q = target.indexOf('?')
  if (q === -1) return { path: target, search: '' }
  return { path: target.slice(0, q), search: target.slice(q) }
}
