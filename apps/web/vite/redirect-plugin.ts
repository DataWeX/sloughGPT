// Vite middleware: 307 legacy-path redirects from lib/redirects.ts (proxy.ts parity).

import type { Plugin } from 'vite'
import type { ServerResponse } from 'node:http'
import { isIgnoredPath, REDIRECTS } from '../lib/redirects'

export function applyRedirect(pathname: string, res: ServerResponse): boolean {
  if (isIgnoredPath(pathname)) return false
  const target = REDIRECTS[pathname]
  if (!target) return false
  res.statusCode = 307
  res.setHeader('location', target)
  res.setHeader('x-vite-redirect', '1')
  res.end()
  return true
}

export function redirectsPlugin(): Plugin {
  return {
    name: 'vite:legacy-redirects',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        try {
          const pathname = (req.url || '/').split('?')[0] || '/'
          // Leave /api/* to apiRoutesPlugin (no overlap with REDIRECTS keys)
          if (pathname.startsWith('/api/')) return next()
          if (applyRedirect(pathname, res)) return
          next()
        } catch (err) {
          next(err as Error)
        }
      })
    },
  }
}
