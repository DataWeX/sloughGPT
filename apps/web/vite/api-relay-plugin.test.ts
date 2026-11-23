import { describe, expect, it } from 'vitest'
import { join } from 'node:path'
import { classify } from './api-relay-plugin'

// Gateway kind-rule parity (card 20261123_135845): the SPA and the Python
// API own the same top-level names (/chat, /models, /training), so *path*
// can never decide who answers — a prefix table would shadow one side.
// Request kind decides: navigations send Accept: text/html (SPA shell via
// vite), the app's own traffic (fetch/XHR/SSE/media) sends */* , json or
// event-stream and must always reach Python. Writes never see the static
// layer (method gate). Mirrors apps/gateway/src/main.rs wants_document +
// build_router's method gate.
const webRoot = join(import.meta.dirname, '..')
const publicDir = join(webRoot, 'public')
const ctx = { root: webRoot, publicDir }

function req(method: string, url: string, headers: Record<string, string> = {}) {
  return { method, url, headers }
}

describe('api-relay-plugin classify', () => {
  it('vite owns real files under root and publicDir (module + asset)', () => {
    expect(classify(req('GET', '/lib/config.ts'), ctx)).toBe('vite')
    // extensionless import URL — the .ts resolution try
    expect(classify(req('GET', '/lib/config'), ctx)).toBe('vite')
    expect(classify(req('GET', '/favicon.svg'), ctx)).toBe('vite')
  })

  it('vite owns its internals, deps and HMR websocket', () => {
    expect(classify(req('GET', '/@vite/client'), ctx)).toBe('vite')
    expect(classify(req('GET', '/node_modules/.vite/deps/react.js'), ctx)).toBe('vite')
    expect(
      classify(req('GET', '/__vite_hmr', { upgrade: 'websocket' }), ctx),
    ).toBe('vite')
  })

  it('navigations (Accept: text/html) stay with vite — the SPA shell', () => {
    expect(classify(req('GET', '/', { accept: 'text/html' }), ctx)).toBe('vite')
    expect(classify(req('GET', '/chat', { accept: 'text/html' }), ctx)).toBe('vite')
    expect(classify(req('GET', '/settings', { accept: 'text/html,*/*' }), ctx)).toBe(
      'vite',
    )
  })

  it('API reads (data accepts) relay to Python', () => {
    expect(classify(req('GET', '/info', { accept: '*/*' }), ctx)).toBe('relay')
    expect(classify(req('GET', '/models/hf', { accept: 'application/json' }), ctx)).toBe(
      'relay',
    )
    expect(
      classify(
        req('GET', '/training/job-events/stream', { accept: 'text/event-stream' }),
        ctx,
      ),
    ).toBe('relay')
    // audio/media: not a file, not a document
    expect(classify(req('GET', '/chat/audio/s1/m1'), ctx)).toBe('relay')
  })

  it('writes relay regardless of accept (method gate, gateway parity)', () => {
    expect(classify(req('POST', '/chat', { accept: 'text/html' }), ctx)).toBe('relay')
    expect(classify(req('PUT', '/session/s1/context'), ctx)).toBe('relay')
    expect(classify(req('DELETE', '/session/s1'), ctx)).toBe('relay')
  })

  it('API doc paths relay even when navigated (gateway API_DOC_PATHS)', () => {
    expect(classify(req('GET', '/docs', { accept: 'text/html' }), ctx)).toBe('relay')
    expect(classify(req('GET', '/redoc', { accept: 'text/html' }), ctx)).toBe('relay')
    expect(classify(req('GET', '/openapi.json'), ctx)).toBe('relay')
  })

  it('never lets traversal escape the root check into vite', () => {
    // normalize resolves .. — not a real file under root → falls through
    // to relay (the API 404s it; vite never sees an escape)
    expect(classify(req('GET', '/../../etc/passwd', { accept: '*/*' }), ctx)).toBe(
      'relay',
    )
  })
})
