import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

const IGNORE_PATHS = ['/_next', '/favicon', '/sw.js', '/workbox']
const SENSITIVE_HEADERS = ['authorization', 'cookie', 'x-api-key']

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`
}

function sanitizeHeaders(headers: Headers): Record<string, string> {
  const sanitized: Record<string, string> = {}
  headers.forEach((value, key) => {
    if (SENSITIVE_HEADERS.includes(key.toLowerCase())) {
      sanitized[key] = '[REDACTED]'
    } else {
      sanitized[key] = value
    }
  })
  return sanitized
}

export function middleware(request: NextRequest) {
  const start = Date.now()
  const { method, nextUrl, headers } = request

  const ignorePath = IGNORE_PATHS.some(path => nextUrl.pathname.startsWith(path))
  if (ignorePath) {
    return NextResponse.next()
  }

  const response = NextResponse.next()
  const duration = Date.now() - start
  const status = response.status

  const level = status >= 500 ? 'ERROR' : status >= 400 ? 'WARN' : 'INFO'
  const methodColor = method === 'GET' ? '\x1b[36m' : method === 'POST' ? '\x1b[33m' : '\x1b[35m'
  const statusColor = status >= 500 ? '\x1b[31m' : status >= 400 ? '\x1b[33m' : '\x1b[32m'
  const reset = '\x1b[0m'

  const userAgent = headers.get('user-agent') || 'unknown'
  const clientIp = headers.get('x-forwarded-for') || headers.get('x-real-ip') || '-'

  const logLine = [
    `[${level}]`,
    `${methodColor}${method}${reset}`,
    `${nextUrl.pathname}`,
    `${statusColor}${status}${reset}`,
    `${duration}ms`,
  ].join(' ')

  console.log(logLine)

  response.headers.set('X-Response-Time', `${duration}ms`)
  response.headers.set('X-Request-ID', crypto.randomUUID())

  return response
}

export const config = {
  matcher: [
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
}
