// Vite plugin: same-origin dev — API traffic relays to Python, everything
// else stays with vite (card 20261123_135845: one browser-facing port).
//
// The dev face of the gateway's one-origin rule (apps/gateway/src/main.rs):
// the SPA and the Python API own the same top-level names (/chat, /models,
// /training — 600+ root paths), so *path* can never decide who answers — a
// prefix table would shadow one side. Request KIND decides (wants_document):
// navigations send `Accept: text/html` and get the SPA shell; the app's own
// traffic (fetch/XHR/SSE/media) sends `*/*`, json or event-stream and must
// always reach Python; writes never see the static layer (method gate).
//
// Streaming-safe by construction — raw node:http pipe, no buffering, no
// total timeout: POST /chat's SSE and every event-stream ride through
// untouched (the gateway's relay contract: "relays response bytes untouched
// — no buffering, no JSON parsing, no SSE re-chunking").
//
// Target: SLO_API_URL (default http://127.0.0.1:8000 — the canonical API
// port; machines where a squatter holds it run the API elsewhere and set
// SLO_API_URL=http://127.0.0.1:8001).
//
// Register AFTER apiRoutesPlugin: /api/(planner|calendar) dev handlers run
// first and next() out of scope before this relay ever sees a request.

import type { Plugin } from 'vite'
import type { IncomingMessage, ServerResponse } from 'node:http'
import http from 'node:http'
import { statSync } from 'node:fs'
import { join, normalize } from 'node:path'

export type RelayDecision = 'vite' | 'relay'

export interface ClassifyRequest {
  method?: string
  url?: string
  headers: Record<string, string | string[] | undefined>
}

export interface ClassifyContext {
  root: string
  publicDir: string
}

/** Gateway API_DOC_PATHS — the docs are Python's, even as navigations. */
const API_DOC_PATHS = ['/docs', '/redoc', '/openapi.json']

/** Vite-owned URL space: transforms, prebundled deps, HMR. */
const VITE_PREFIXES = ['/@', '/node_modules', '/__vite']

/** Extensions tried for extensionless module URLs (`/lib/config`). */
const FILE_EXTS = ['', '.ts', '.tsx', '.js', '.jsx', '.mjs', '.json', '.css']

/** Hop-by-hop headers neither side of the pipe may forward. */
const HOP_BY_HOP = new Set([
  'connection',
  'keep-alive',
  'transfer-encoding',
  'upgrade',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailers',
])

function header(req: ClassifyRequest, name: string): string | undefined {
  const v = req.headers[name]
  return Array.isArray(v) ? v.join(', ') : v
}

/** True when the client asks for the SPA shell rather than data. */
function wantsDocument(req: ClassifyRequest): boolean {
  return (header(req, 'accept') ?? '').includes('text/html')
}

/** A real file under root or publicDir — vite serves it (never the relay). */
function isViteFile(pathname: string, ctx: ClassifyContext): boolean {
  const rel = normalize(pathname).replace(/^[/\\]+/, '')
  if (!rel || rel.startsWith('..')) return false // traversal never escapes
  for (const base of [ctx.root, ctx.publicDir]) {
    for (const ext of FILE_EXTS) {
      try {
        if (statSync(join(base, rel + ext)).isFile()) return true
      } catch {
        /* not this candidate */
      }
    }
  }
  return false
}

/**
 * Dev twin of the gateway's routing rule — kind over path, always.
 * Returns whose middleware answers: `vite` (pass through) or `relay`
 * (handle here, never forwarded further).
 */
export function classify(req: ClassifyRequest, ctx: ClassifyContext): RelayDecision {
  // HMR websocket — vite owns upgrades (the gateway has no dev sockets).
  const upgrade = header(req, 'upgrade')
  if (upgrade && upgrade.toLowerCase().includes('websocket')) return 'vite'

  // Method gate: writes never see the static layer (gateway parity).
  const method = (req.method ?? 'GET').toUpperCase()
  if (method !== 'GET' && method !== 'HEAD') return 'relay'

  const pathname = (req.url ?? '/').split('?')[0] || '/'

  if (VITE_PREFIXES.some((p) => pathname.startsWith(p))) return 'vite'
  if (isViteFile(pathname, ctx)) return 'vite'

  // /docs, /redoc, /openapi.json — Python answers even document requests.
  if (API_DOC_PATHS.some((p) => pathname === p || pathname.startsWith(`${p}/`))) {
    return 'relay'
  }

  return wantsDocument(req) ? 'vite' : 'relay'
}

/** Byte-relay one request to `target` — streaming, unbuffered, untimed. */
function relay(target: string, req: IncomingMessage, res: ServerResponse): void {
  const url = new URL(target)
  const headers: Record<string, string | string[] | undefined> = {
    ...req.headers,
    host: url.host,
  }
  for (const h of HOP_BY_HOP) delete headers[h]

  const upstream = http.request(
    {
      hostname: url.hostname,
      port: url.port || undefined,
      path: req.url,
      method: req.method,
      headers,
    },
    (up) => {
      const out: Record<string, string | string[]> = {}
      for (const [k, v] of Object.entries(up.headers)) {
        if (HOP_BY_HOP.has(k) || v === undefined) continue
        out[k] = v
      }
      res.writeHead(up.statusCode ?? 502, out)
      up.pipe(res) // SSE rides here — no buffering, no re-chunking
    },
  )
  upstream.on('error', (err) => {
    if (!res.headersSent) {
      res.writeHead(502, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ error: `api relay (${target}): ${err.message}` }))
    } else {
      res.end()
    }
  })
  req.pipe(upstream)
}

export function apiRelayPlugin(): Plugin {
  let ctx: ClassifyContext = { root: '', publicDir: '' }
  let target = 'http://127.0.0.1:8000'

  return {
    name: 'vite:api-relay',
    configResolved(config) {
      ctx = { root: config.root, publicDir: config.publicDir }
      target = process.env.SLO_API_URL || 'http://127.0.0.1:8000'
    },
    configureServer(server) {
      ctx = { root: server.config.root, publicDir: server.config.publicDir }
      server.middlewares.use((req, res, next) => {
        const decision = classify(
          { method: req.method, url: req.url, headers: req.headers },
          ctx,
        )
        if (decision === 'vite') return next()
        relay(target, req, res)
      })
    },
  }
}
