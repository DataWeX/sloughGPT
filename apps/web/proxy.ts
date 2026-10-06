import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'
// Single source of truth — redirects.ts is shared by this proxy and the
// Vite dev/static middleware. An inline copy here silently drifted (37 vs
// 47: the ten /workspace legacy paths were missing), so import, don't fork.
import { IGNORE_PATHS, REDIRECTS } from './lib/redirects'

export function proxy(request: NextRequest) {
  const start = Date.now()
  const { method, nextUrl, headers } = request

  const ignorePath = IGNORE_PATHS.some((path) => nextUrl.pathname.startsWith(path))
  if (ignorePath) {
    return NextResponse.next()
  }

  const redirectTarget = REDIRECTS[nextUrl.pathname]
  if (redirectTarget) {
    return NextResponse.redirect(new URL(redirectTarget, request.url))
  }

  const response = NextResponse.next()
  const duration = Date.now() - start
  const status = response.status

  const level = status >= 500 ? 'ERROR' : status >= 400 ? 'WARN' : 'INFO'
  const methodColor = method === 'GET' ? '\x1b[36m' : method === 'POST' ? '\x1b[33m' : '\x1b[35m'
  const statusColor = status >= 500 ? '\x1b[31m' : status >= 400 ? '\x1b[33m' : '\x1b[32m'
  const reset = '\x1b[0m'

  const logLine = [
    `[${level}]`,
    `${methodColor}${method}${reset}`,
    `${nextUrl.pathname}`,
    `${statusColor}${status}${reset}`,
    `${duration}ms`,
  ].join(' ')

  if (process.env.NODE_ENV === 'development') {
    console.log(logLine)
  }

  response.headers.set('X-Response-Time', `${duration}ms`)
  response.headers.set('X-Request-ID', crypto.randomUUID())

  return response
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
}
