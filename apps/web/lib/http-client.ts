export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: unknown,
    public requestId?: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}
import { logger } from './dev-log'

import { PUBLIC_API_URL } from './config'
import { useAuthStore } from './auth'

// ── Correlation ID propagation ───────────────────────────────────────
// Generate a short UUID for each request so the frontend can correlate
// with server-side logs via `X-Correlation-ID`. The server reads this
// header and includes it in every log line.

function _corrId(): string {
  // 8-char hex ID matching server-side format
  return Math.random().toString(16).slice(2, 10)
}

// Store last N correlation IDs for debugging
const _recentCorrIds: Array<{ id: string; url: string; ts: number }> = []
const MAX_RECENT = 20

/** Get the last N correlation IDs for debugging. */
export function getRecentCorrelationIds(): Array<{ id: string; url: string; ts: number }> {
  return _recentCorrIds.slice()
}

export interface RequestOptions {
  signal?: AbortSignal
  timeout?: number
  noAuth?: boolean
  raw?: boolean
  onProgress?: (pct: number) => void
  headers?: Record<string, string>
  silent?: boolean
}

const RETRYABLE_STATUSES = new Set([408, 429, 502, 503, 504])
const MAX_RETRIES = 2
const BASE_DELAY = 500
const DEFAULT_TIMEOUT_MS = 30_000

async function request<T>(
  method: string,
  url: string,
  body?: unknown,
  opts?: RequestOptions,
): Promise<T> {
  const apiUrl = `${PUBLIC_API_URL}${url}`
  const corrId = _corrId()
  const headers: Record<string, string> = {}
  if (!opts?.raw) headers['Content-Type'] = 'application/json'
  if (opts?.headers) Object.assign(headers, opts.headers)
  if (!opts?.noAuth) {
    const token = useAuthStore.getState().token
    if (token) headers['Authorization'] = `Bearer ${token}`
  }
  headers['X-Correlation-ID'] = corrId

  // Track for debugging
  _recentCorrIds.push({ id: corrId, url, ts: Date.now() })
  if (_recentCorrIds.length > MAX_RECENT) _recentCorrIds.shift()

  logger.debug(`>>> ${method} ${url} corr=${corrId}`, { corrId, method, url })

  const timeoutMs = opts?.timeout ?? DEFAULT_TIMEOUT_MS
  let retries = 0
  while (true) {
    let signal = opts?.signal
    let timer: ReturnType<typeof setTimeout> | undefined
    if (!signal) {
      const ac = new AbortController()
      timer = setTimeout(() => ac.abort(), timeoutMs)
      signal = ac.signal
    }
    try {
      const res = await fetch(apiUrl, {
        method,
        headers,
        body: opts?.raw ? (body as BodyInit) : body != null ? JSON.stringify(body) : undefined,
        signal,
      })
      if (timer) clearTimeout(timer)

      logger.debug(`<<< ${method} ${url} ${res.status} corr=${corrId}`, { corrId, method, url, status: res.status })

      if (!res.ok) {
        const status = res.status
        const isRetryable = RETRYABLE_STATUSES.has(status)
        if (isRetryable && retries < MAX_RETRIES) {
          retries++
          const retryAfter = Number(res.headers.get('Retry-After')) || 0
          const delay = retryAfter > 0 ? retryAfter * 1000 : BASE_DELAY * Math.pow(2, retries - 1)
          await new Promise(r => setTimeout(r, delay))
          continue
        }

        const text = await res.text()
        const requestId = res.headers.get('X-Request-ID') || undefined
        let detail: string | undefined
        try { const j = JSON.parse(text); detail = j.detail ?? j.message ?? j.error }
        catch { detail = text || res.statusText }
        const message = Array.isArray(detail) ? detail.map((d: { msg?: string } | string) => typeof d === 'string' ? d : d.msg ?? '').join('; ') : detail || 'Could not request'

        if (!opts?.silent) {
          const apiErr = new ApiError(message, status, { raw: text }, requestId)
          import('./error-store').then(({ useErrorStore }) => {
            useErrorStore.getState().addError(apiErr, {
              source: url,
              title: status >= 500 ? 'Server Error' : status >= 400 ? `HTTP ${status}` : 'API Error',
              requestId: requestId,
            })
          })
        }
        throw new ApiError(message, status, { raw: text }, requestId)
      }

      const text = await res.text()
      if (!text) return undefined as T
      const json = JSON.parse(text)
      // Unwrap StandardResponse envelope: {status, data, message, meta} → data
      if (json && typeof json === 'object' && 'status' in json && 'data' in json) {
        // Attach meta as non-enumerable so callers can access it if needed
        const result = json.data as T
        if (json.meta && typeof result === 'object' && result !== null) {
          Object.defineProperty(result, '_meta', {value: json.meta, enumerable: false})
        }
        return result
      }
      return json as T
    } catch (e: unknown) {
      if (timer) clearTimeout(timer)
      if (e instanceof ApiError) throw e
      // A caller-provided abort is terminal: never retry an explicitly cancelled
      // request, and surface the abort instead of relabelling it as a timeout.
      if (opts?.signal?.aborted) throw e
      const status = 0
      const name = e instanceof Error ? e.name : undefined
      const message_ = e instanceof Error ? e.message : undefined
      const cause = e instanceof Error ? (e as Error & { cause?: { code?: string } }).cause : undefined
      const isTimeout = name === 'AbortError' || message_?.includes('aborted')
      const isConnRefused = message_ === 'Failed to fetch' || cause?.code === 'ECONNREFUSED'
      const message = isTimeout
        ? `Request timed out after ${timeoutMs / 1000}s`
        : isConnRefused ? 'Connection unavailable — server may be starting up' : (message_ || 'Could not request')

      const kind = isTimeout ? 'timeout' : isConnRefused ? 'connection_refused' : 'unknown'

      // Only report after retries exhausted (don't flood on each retry)
      if (retries < MAX_RETRIES) {
        retries++
        await new Promise(r => setTimeout(r, BASE_DELAY * Math.pow(2, retries - 1)))
        continue
      }

      if (!opts?.silent) {
        const apiErr = new ApiError(message, status)
        import('./error-store').then(({ useErrorStore }) => {
          useErrorStore.getState().addError(apiErr, {
            source: url,
            title: 'Connection Error',
          })
        })
        import('./api-monitor-store').then(({ useApiMonitor }) => {
          useApiMonitor.getState().addFailure({
            endpoint: url,
            error: message,
            status: 0,
            timeoutMs,
            timestamp: Date.now(),
            kind,
          })
        })
      }

      throw new ApiError(message, status)
    }
  }
}

// In-flight GET dedup: concurrent identical GETs share one network round-trip.
// Skipped when a signal is provided (caller owns cancellation semantics).
const _inflight = new Map<string, Promise<unknown>>()

export async function apiGet<T>(url: string, params?: Record<string, string>, opts?: RequestOptions): Promise<T> {
  const qs = params ? '?' + new URLSearchParams(params).toString() : ''
  const fullUrl = url + qs
  if (!opts?.signal) {
    const existing = _inflight.get(fullUrl)
    if (existing) return existing as Promise<T>
  }
  const p = request<T>('GET', fullUrl, undefined, opts)
  if (!opts?.signal) _inflight.set(fullUrl, p)
  try {
    return await p
  } finally {
    if (!opts?.signal) _inflight.delete(fullUrl)
  }
}

export async function apiPost<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T> {
  return request<T>('POST', url, body, opts)
}

export async function apiPut<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T> {
  return request<T>('PUT', url, body, opts)
}

export async function apiDelete<T>(url: string, opts?: RequestOptions): Promise<T> {
  return request<T>('DELETE', url, undefined, opts)
}

export async function apiPatch<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T> {
  return request<T>('PATCH', url, body, opts)
}

interface ApiClientConfig {
  signal?: AbortSignal
  params?: Record<string, string>
  _silent?: boolean
  _noAuth?: boolean
  _raw?: boolean
}

function createApiClient(baseURL?: string) {
  const prefix = baseURL ?? PUBLIC_API_URL
  return {
    defaults: { baseURL: prefix },
    get: <T>(url: string, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiGet<T>(u, config?.params, { signal: config?.signal, silent: config?._silent, noAuth: config?._noAuth })
    },
    post: <T>(url: string, body?: unknown, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiPost<T>(u, body, { signal: config?.signal, silent: config?._silent, noAuth: config?._noAuth, raw: config?._raw })
    },
    put: <T>(url: string, body?: unknown, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiPut<T>(u, body, { signal: config?.signal, silent: config?._silent, noAuth: config?._noAuth })
    },
    delete: <T>(url: string, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiDelete<T>(u, { signal: config?.signal, silent: config?._silent, noAuth: config?._noAuth })
    },
    patch: <T>(url: string, body?: unknown, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiPatch<T>(u, body, { signal: config?.signal, silent: config?._silent, noAuth: config?._noAuth })
    },
  }
}

export const apiClient = createApiClient()
export { createApiClient }

// ── Shared fetch helpers ──────────────────────────────────────────────

export interface SSEEvent {
  stream?: string
  phase?: string
  status?: string
  data?: Record<string, unknown>
  meta?: Record<string, unknown>
  message?: string
  error?: string
  id?: string
}

interface AuthFetchOptions extends RequestInit {
  noAuth?: boolean
}

/**
 * Fetch with automatic auth token injection. Use for blob downloads and
 * SSE streaming where the shared request() helper's JSON parsing is
 * unsuitable.
 */
export async function authFetch(url: string, opts?: AuthFetchOptions): Promise<Response> {
  const corrId = _corrId()
  const headers: Record<string, string> = {}
  if (!opts?.noAuth) {
    const token = useAuthStore.getState().token
    if (token) headers['Authorization'] = `Bearer ${token}`
  }
  if (opts?.headers) {
    if (opts.headers instanceof Headers) {
      opts.headers.forEach((v, k) => { headers[k] = v })
    } else {
      Object.assign(headers, opts.headers)
    }
  }
  headers['X-Correlation-ID'] = corrId
  _recentCorrIds.push({ id: corrId, url, ts: Date.now() })
  if (_recentCorrIds.length > MAX_RECENT) _recentCorrIds.shift()
  return fetch(url, { ...opts, headers })
}

interface StreamSSEOptions {
  method?: string
  body?: unknown
  signal?: AbortSignal
  noAuth?: boolean
  lastEventId?: string
}

/**
 * Connect to an SSE endpoint and yield parsed events. Handles the
 * fetch → reader → decode → split → parse lifecycle shared by all
 * streaming controllers.
 */
export async function* streamSSE(url: string, opts?: StreamSSEOptions): AsyncGenerator<SSEEvent> {
  const method = opts?.method ?? 'POST'
  const corrId = _corrId()
  const headers: Record<string, string> = {}
  if (method !== 'GET') headers['Content-Type'] = 'application/json'
  if (!opts?.noAuth) {
    const token = useAuthStore.getState().token
    if (token) headers['Authorization'] = `Bearer ${token}`
  }
  if (opts?.lastEventId) {
    headers['Last-Event-ID'] = opts.lastEventId
  }
  headers['X-Correlation-ID'] = corrId

  _recentCorrIds.push({ id: corrId, url, ts: Date.now() })
  if (_recentCorrIds.length > MAX_RECENT) _recentCorrIds.shift()

  logger.debug(`>>> SSE ${method} ${url} corr=${corrId}`, { corrId, method, url })

  let res: Response
  try {
    res = await fetch(`${PUBLIC_API_URL}${url}`, {
      method,
      headers,
      body: opts?.body != null ? JSON.stringify(opts.body) : undefined,
      signal: opts?.signal,
    })
  } catch (e) {
    const msg = e instanceof Error ? e.message : 'Network error'
    logger.error(`<<< SSE ${method} ${url} FAILED corr=${corrId}: ${msg}`, { corrId })
    yield { status: 'error', message: `Connection error: ${msg}` }
    return
  }

  logger.debug(`<<< SSE ${method} ${url} ${res.status} corr=${corrId}`, { corrId, status: res.status })

  if (!res.ok || !res.body) {
    yield { status: 'error', message: `HTTP ${res.status}${res.statusText ? `: ${res.statusText}` : ''}`, data: { http_status: res.status, error: `HTTP ${res.status}` } }
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      let chunk: ReadableStreamReadResult<Uint8Array>
      try {
        chunk = await reader.read()
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Read error'
        yield { status: 'error', message: `Stream disconnected: ${msg}` }
        return
      }
      if (chunk.done) break
      buffer += decoder.decode(chunk.value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        const trimmed = line.trimEnd()
        if (!trimmed.startsWith('data:')) continue
        const payload = trimmed.slice(5).trim()
        if (!payload || payload === '[DONE]') continue
        try {
          yield JSON.parse(payload) as SSEEvent
        } catch (e) {
          logger.warning('SSE malformed JSON payload skipped', { payload: payload.slice(0, 80), exception: String(e) })
        }
      }
    }
    // Drain remaining buffer
    if (buffer.startsWith('data:')) {
      const payload = buffer.slice(5).trim()
      if (payload && payload !== '[DONE]') {
        try { yield JSON.parse(payload) as SSEEvent } catch { /* skip */ }
      }
    }
  } finally {
    reader.releaseLock()
  }
}
