/**
 * Compare current pathname to a nav `href` when Next.js uses `trailingSlash: true`
 * (`/chat/` vs `/chat`). Also normalizes root.
 * Supports prefix matching so nested routes (e.g. /training/job/xyz) highlight parent.
 */
export function routeMatchesPath(pathname: string, href: string): boolean {
  const norm = (p: string) => {
    const t = p.trim()
    if (t === '' || t === '/') return '/'
    return t.replace(/\/+$/, '')
  }
  const a = norm(pathname)
  const b = norm(href)
  if (a === b) return true
  // Prefix match: /training/job/xyz matches /training
  // But /chat/sessions must not match /channels
  return a.startsWith(b + '/')
}
