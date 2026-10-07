// Vite plugin: serve app/api/planner + app/api/calendar route handlers in dev.
// Uses ssrLoadModule so helpers.ts can use Node fs; aliases next/server via vite.config.

import type { Plugin } from 'vite'
import type { IncomingMessage, ServerResponse } from 'node:http'
import {
  discoverApiRoutes,
  isScopedApiPath,
  matchApiRoute,
  type ApiRouteDef,
} from './api-routes.ts'

type RouteModule = Record<string, unknown>

type Handler = (
  request: Request,
  ctx?: { params?: Promise<Record<string, string>> },
) => Promise<Response> | Response

async function readBody(req: IncomingMessage): Promise<Buffer | undefined> {
  const method = (req.method || 'GET').toUpperCase()
  if (method === 'GET' || method === 'HEAD') return undefined
  const chunks: Buffer[] = []
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
  }
  if (chunks.length === 0) return undefined
  return Buffer.concat(chunks)
}

function toRequest(req: IncomingMessage, body?: Buffer): Request {
  const host = req.headers.host || '127.0.0.1:5173'
  const url = `http://${host}${req.url || '/'}`
  const headers = new Headers()
  for (const [key, value] of Object.entries(req.headers)) {
    if (value === undefined) continue
    if (Array.isArray(value)) {
      for (const v of value) headers.append(key, v)
    } else {
      headers.set(key, String(value))
    }
  }
  const init: RequestInit & { duplex?: 'half' } = {
    method: req.method || 'GET',
    headers,
  }
  if (body) {
    init.body = new Uint8Array(body)
    init.duplex = 'half'
  }
  return new Request(url, init)
}

async function send(res: ServerResponse, response: Response): Promise<void> {
  res.statusCode = response.status
  response.headers.forEach((value, key) => {
    if (key === 'transfer-encoding') return
    res.setHeader(key, value)
  })
  const buf = Buffer.from(await response.arrayBuffer())
  res.end(buf)
}

export async function invokeRouteHandler(
  mod: RouteModule,
  method: string,
  request: Request,
  params: Record<string, string>,
): Promise<Response> {
  const handler = mod[method] as Handler | undefined
  if (typeof handler !== 'function') {
    return Response.json({ error: `Method ${method} not allowed` }, { status: 405 })
  }
  return handler(request, { params: Promise.resolve(params) })
}

/** Convert absolute route file path to Vite ssrLoadModule URL relative to root. */
export function toSsrUrl(file: string, root: string): string {
  const normalizedRoot = root.replace(/\\/g, '/').replace(/\/$/, '')
  const normalizedFile = file.replace(/\\/g, '/')
  if (normalizedFile.startsWith(normalizedRoot + '/')) {
    return normalizedFile.slice(normalizedRoot.length)
  }
  const idx = normalizedFile.indexOf('/apps/web/')
  if (idx !== -1) return normalizedFile.slice(idx + '/apps/web'.length)
  return normalizedFile
}

export function apiRoutesPlugin(): Plugin {
  let routes: ApiRouteDef[] = []
  let root = ''

  return {
    name: 'vite:api-routes',
    configResolved(config) {
      root = config.root
      routes = discoverApiRoutes(root)
    },
    handleHotUpdate(ctx) {
      if (ctx.file.includes('/app/api/planner/') || ctx.file.includes('/app/api/calendar/')) {
        routes = discoverApiRoutes(root)
      }
    },
    configureServer(server) {
      root = server.config.root
      routes = discoverApiRoutes(root)
      server.middlewares.use(async (req, res, next) => {
        try {
          const rawUrl = req.url || '/'
          const pathname = rawUrl.split('?')[0] || '/'
          if (!isScopedApiPath(pathname)) return next()

          const matched = matchApiRoute(pathname, routes)
          if (!matched) {
            res.statusCode = 404
            res.setHeader('content-type', 'application/json')
            res.end(JSON.stringify({ error: 'Not found' }))
            return
          }

          const method = (req.method || 'GET').toUpperCase()
          if (!matched.route.methods.includes(method)) {
            res.statusCode = 405
            res.setHeader('content-type', 'application/json')
            res.end(JSON.stringify({ error: `Method ${method} not allowed` }))
            return
          }

          const body = await readBody(req)
          const request = toRequest(req, body)
          const ssrUrl = toSsrUrl(matched.route.file, server.config.root)
          const mod = (await server.ssrLoadModule(ssrUrl)) as RouteModule
          const response = await invokeRouteHandler(mod, method, request, matched.params)
          await send(res, response)
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err)
          if (!res.headersSent) {
            res.statusCode = 500
            res.setHeader('content-type', 'application/json')
            res.end(JSON.stringify({ error: message }))
          } else {
            next(err)
          }
        }
      })
    },
  }
}
