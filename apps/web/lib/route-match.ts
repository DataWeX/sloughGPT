/**
 * Compare current pathname to a nav `href` when Next.js uses `trailingSlash: true`
 * (`/chat/` vs `/chat`). Also normalizes root.
 */
export function routeMatchesPath(pathname: string, href: string): boolean {
  const norm = (p: string) => {
    const t = p.trim()
    if (t === '' || t === '/') return '/'
    return t.replace(/\/+$/, '')
  }
  return norm(pathname) === norm(href)
}
