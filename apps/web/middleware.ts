import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

const IGNORE_PATHS = ['/_next', '/favicon', '/sw.js', '/workbox']

const REDIRECTS: Record<string, string> = {
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
  '/multimodal': '/models',
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
  '/voice': '/developer',
  '/shell': '/developer',
  '/vm': '/developer',
  '/workflow': '/feedback',
}

export function middleware(request: NextRequest) {
  const start = Date.now()
  const { method, nextUrl, headers } = request

  const ignorePath = IGNORE_PATHS.some(path => nextUrl.pathname.startsWith(path))
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
  matcher: [
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
}
