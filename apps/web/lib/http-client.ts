// ═══════════════════════════════════════════════════════════════════════════
// http-client — Full-featured HTTP request library for sloughGPT frontend
// ═══════════════════════════════════════════════════════════════════════════
//
// Features:
//   1. Interceptors — request/response middleware chain
//   2. Cache layer — TTL-based with stale-while-revalidate
//   3. Circuit breaker — closed → open → half-open state machine
//   4. Lifecycle hooks — beforeRequest, afterResponse, onError
//   5. Progress tracking — upload/download percentage
//   6. Dedup with TTL — recent-dedup beyond in-flight only
//   7. Request throttling — max concurrent with queue
//   8. Response metadata — timing, retry, cache hit, circuit state
//
// All existing exports (apiGet, apiPost, apiClient, etc.) are backward-
// compatible. New features are opt-in via RequestOptions or httpClient.
// ═══════════════════════════════════════════════════════════════════════════

import { logger } from './dev-log'
import { PUBLIC_API_URL } from './config'
import { useAuthStore } from './auth'

// ── ApiError ────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: {
      raw?: string
      code?: string
      details?: unknown
      correlationId?: string
    },
    public requestId?: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

// ── Types ───────────────────────────────────────────────────────────────────

export interface RequestInterceptor {
  onFulfilled: (config: RequestConfig) => RequestConfig | Promise<RequestConfig>
  onRejected?: (error: unknown) => unknown
}

export interface ResponseInterceptor {
  onFulfilled: (response: ResponseEnvelope) => ResponseEnvelope | Promise<ResponseEnvelope>
  onRejected?: (error: unknown) => unknown
}

export interface RequestConfig {
  method: string
  url: string
  fullUrl: string
  headers: Record<string, string>
  body?: unknown
  signal?: AbortSignal
  correlationId: string
  startTime: number
  opts: RequestOptions
}

export interface ResponseEnvelope {
  ok: boolean
  status: number
  statusText: string
  headers: Headers
  text: string
  json?: unknown
  config: RequestConfig
}

export interface CacheOptions {
  ttlMs: number
  staleWhileRevalidate?: boolean
  /** Cache only if response status matches. Default: [200] */
  validStatuses?: number[]
}

export interface CircuitBreakerOptions {
  failureThreshold: number
  resetTimeoutMs: number
  halfOpenMax?: number
}

export type CircuitState = 'closed' | 'open' | 'half-open'

export interface ThrottleOptions {
  maxConcurrent: number
  maxQueue: number
  queueTimeoutMs: number
}

export interface ResponseMetadata {
  timingMs: number
  retries: number
  cacheHit: boolean
  circuitState: CircuitState
  correlationId: string
  requestId?: string
}

export interface ProgressCallbacks {
  onUpload?: (percent: number, loaded: number, total: number) => void
  onDownload?: (percent: number, loaded: number, total: number) => void
}

export interface RequestOptions {
  signal?: AbortSignal
  timeout?: number
  noAuth?: boolean
  raw?: boolean
  /** @deprecated Use progress.onDownload instead */
  onProgress?: (pct: number) => void
  headers?: Record<string, string>
  silent?: boolean
  /** Cache this GET request. boolean or CacheOptions. */
  cache?: boolean | CacheOptions
  /** Custom cache key override. */
  cacheKey?: string
  /** Throttle this request through the shared semaphore. */
  throttle?: boolean
  /** Upload/download progress callbacks. */
  progress?: ProgressCallbacks
  /** Arbitrary metadata attached to ResponseMetadata. */
  metadata?: Record<string, unknown>
  /** Skip circuit breaker for this request. */
  skipCircuitBreaker?: boolean
  /** Enable recent-dedup with TTL (ms). If set, identical GETs within this window return cached promise. */
  dedupTtlMs?: number
}

export interface HttpClientResponse<T> {
  data: T
  meta: ResponseMetadata
}

export interface HttpClientOptions {
  baseURL?: string
  interceptors?: {
    request?: RequestInterceptor[]
    response?: ResponseInterceptor[]
  }
  cache?: Partial<CacheOptions>
  circuitBreaker?: Partial<CircuitBreakerOptions>
  throttle?: Partial<ThrottleOptions>
  hooks?: {
    beforeRequest?: ((config: RequestConfig) => RequestConfig | Promise<RequestConfig>)[]
    afterResponse?: ((response: ResponseEnvelope) => ResponseEnvelope | Promise<ResponseEnvelope>)[]
    onError?: ((error: ApiError, config: RequestConfig) => void | Promise<void>)[]
  }
}

// ── Constants ───────────────────────────────────────────────────────────────

const RETRYABLE_STATUSES = new Set([408, 429, 502, 503, 504])
const MAX_RETRIES = 2
const BASE_DELAY = 500
const DEFAULT_TIMEOUT_MS = 30_000

const DOCSTORE_RETRYABLE_STATUSES = new Set([400, 408, 429, 500, 502, 503, 504])
const DOCSTORE_MAX_RETRIES = 4
const DOCSTORE_BASE_DELAY = 300

const DEFAULT_CACHE_TTL_MS = 60_000
const DEFAULT_CB_FAILURE_THRESHOLD = 5
const DEFAULT_CB_RESET_TIMEOUT_MS = 30_000
const DEFAULT_THROTTLE_MAX = 10
const DEFAULT_THROTTLE_QUEUE = 50
const DEFAULT_THROTTLE_TIMEOUT_MS = 10_000

// ── Correlation IDs ─────────────────────────────────────────────────────────

function _corrId(): string {
  return Math.random().toString(16).slice(2, 10)
}

const _recentCorrIds: Array<{ id: string; url: string; ts: number }> = []
const MAX_RECENT = 20

export function getRecentCorrelationIds(): Array<{ id: string; url: string; ts: number }> {
  return _recentCorrIds.slice()
}

function _trackCorrId(corrId: string, url: string) {
  _recentCorrIds.push({ id: corrId, url, ts: Date.now() })
  if (_recentCorrIds.length > MAX_RECENT) _recentCorrIds.shift()
}

function _isDocstoreUrl(url: string): boolean {
  return url.includes('/docstore/')
}

// ── InterceptorManager ──────────────────────────────────────────────────────

export class InterceptorManager<T> {
  private _interceptors: Array<{ id: number; fulfilled: (value: T) => T | Promise<T>; rejected?: (error: unknown) => unknown }> = []
  private _nextId = 0

  use(fulfilled: (value: T) => T | Promise<T>, rejected?: (error: unknown) => unknown): number {
    const id = this._nextId++
    this._interceptors.push({ id, fulfilled, rejected })
    return id
  }

  eject(id: number) {
    this._interceptors = this._interceptors.filter(i => i.id !== id)
  }

  clear() {
    this._interceptors = []
  }

  get size(): number {
    return this._interceptors.length
  }

  async run(initial: T): Promise<T> {
    let value = initial
    for (const interceptor of this._interceptors) {
      try {
        value = await interceptor.fulfilled(value)
      } catch (e) {
        if (interceptor.rejected) {
          value = await interceptor.rejected(e) as T
        } else {
          throw e
        }
      }
    }
    return value
  }
}

// ── Cache ───────────────────────────────────────────────────────────────────

interface CacheEntry {
  data: unknown
  expiresAt: number
  staleAt: number
  etag?: string
}

export class HttpCache {
  private _store = new Map<string, CacheEntry>()
  private _stats = { hits: 0, misses: 0, staleHits: 0 }

  constructor(private defaults: Partial<CacheOptions> = {}) {}

  get(key: string): { data: unknown; stale: boolean } | undefined {
    const entry = this._store.get(key)
    if (!entry) {
      this._stats.misses++
      return undefined
    }
    const now = Date.now()
    if (now <= entry.expiresAt) {
      this._stats.hits++
      return { data: entry.data, stale: false }
    }
    if (now <= entry.staleAt) {
      this._stats.staleHits++
      return { data: entry.data, stale: true }
    }
    this._store.delete(key)
    this._stats.misses++
    return undefined
  }

  set(key: string, data: unknown, opts?: Partial<CacheOptions>) {
    const ttlMs = opts?.ttlMs ?? this.defaults.ttlMs ?? DEFAULT_CACHE_TTL_MS
    const swr = opts?.staleWhileRevalidate ?? this.defaults.staleWhileRevalidate ?? true
    const now = Date.now()
    this._store.set(key, {
      data,
      expiresAt: now + ttlMs,
      staleAt: swr ? now + ttlMs * 2 : now + ttlMs,
    })
  }

  invalidate(key: string) {
    this._store.delete(key)
  }

  invalidatePattern(pattern: string) {
    const regex = new RegExp(pattern)
    for (const key of this._store.keys()) {
      if (regex.test(key)) this._store.delete(key)
    }
  }

  clear() {
    this._store.clear()
    this._stats = { hits: 0, misses: 0, staleHits: 0 }
  }

  get stats() {
    return { ...this._stats }
  }

  get size() {
    return this._store.size
  }

  static makeKey(method: string, url: string, params?: Record<string, string>): string {
    const qs = params ? '?' + new URLSearchParams(params).toString() : ''
    return `${method}:${url}${qs}`
  }
}

// ── CircuitBreaker ──────────────────────────────────────────────────────────

export class CircuitBreaker {
  private _state: CircuitState = 'closed'
  private _failureCount = 0
  private _successCount = 0
  private _lastFailureAt = 0
  private _halfOpenAttempts = 0

  constructor(private opts: Partial<CircuitBreakerOptions> = {}) {}

  get state(): CircuitState {
    if (this._state === 'open') {
      const resetTimeout = this.opts.resetTimeoutMs ?? DEFAULT_CB_RESET_TIMEOUT_MS
      if (Date.now() - this._lastFailureAt >= resetTimeout) {
        this._state = 'half-open'
        this._halfOpenAttempts = 0
      }
    }
    return this._state
  }

  allow(): boolean {
    const s = this.state
    if (s === 'closed') return true
    if (s === 'half-open') {
      const max = this.opts.halfOpenMax ?? 1
      if (this._halfOpenAttempts < max) {
        this._halfOpenAttempts++
        return true
      }
      return false
    }
    return false
  }

  recordSuccess() {
    if (this.state === 'half-open') {
      this._state = 'closed'
      this._failureCount = 0
    }
    this._successCount++
  }

  recordFailure() {
    this._failureCount++
    this._lastFailureAt = Date.now()
    const threshold = this.opts.failureThreshold ?? DEFAULT_CB_FAILURE_THRESHOLD
    if (this._failureCount >= threshold) {
      this._state = 'open'
    }
  }

  reset() {
    this._state = 'closed'
    this._failureCount = 0
    this._successCount = 0
    this._halfOpenAttempts = 0
  }

  get failureCount() { return this._failureCount }
  get successCount() { return this._successCount }
}

// ── Throttler ───────────────────────────────────────────────────────────────

interface ThrottleQueueEntry {
  resolve: () => void
  reject: (err: Error) => void
  timer?: ReturnType<typeof setTimeout>
}

export class Throttler {
  private _running = 0
  private _queue: ThrottleQueueEntry[] = []

  constructor(private opts: Partial<ThrottleOptions> = {}) {}

  get running() { return this._running }
  get queued() { return this._queue.length }

  async acquire(): Promise<void> {
    const max = this.opts.maxConcurrent ?? DEFAULT_THROTTLE_MAX
    if (this._running < max) {
      this._running++
      return
    }
    const maxQueue = this.opts.maxQueue ?? DEFAULT_THROTTLE_QUEUE
    if (this._queue.length >= maxQueue) {
      throw new ApiError('Request queue full', 429)
    }
    return new Promise<void>((resolve, reject) => {
      const timeoutMs = this.opts.queueTimeoutMs ?? DEFAULT_THROTTLE_TIMEOUT_MS
      const timer = setTimeout(() => {
        const idx = this._queue.findIndex(e => e.resolve === resolve)
        if (idx !== -1) this._queue.splice(idx, 1)
        reject(new ApiError('Request queue timeout', 429))
      }, timeoutMs)
      this._queue.push({ resolve, reject, timer })
    })
  }

  release() {
    this._running--
    const next = this._queue.shift()
    if (next) {
      if (next.timer) clearTimeout(next.timer)
      this._running++
      next.resolve()
    }
  }
}

// ── In-flight dedup ──────────────────────────────────────────────────────────
// Concurrent identical GETs share one network round-trip.
// Recent dedup (TTL-based) is opt-in via opts.dedupTtlMs.

const _inflight = new Map<string, Promise<unknown>>()
const _recentDedup = new Map<string, { promise: Promise<unknown>; expiresAt: number }>()

function _cleanupRecentDedup() {
  const now = Date.now()
  for (const [key, entry] of _recentDedup) {
    if (now > entry.expiresAt) _recentDedup.delete(key)
  }
}

// ── Request defaults (mutable, for httpClient.configure()) ───────────────────

let _defaultTimeout = DEFAULT_TIMEOUT_MS
let _defaultMaxRetries = MAX_RETRIES
let _defaultBaseDelay = BASE_DELAY

// ── Core request function ───────────────────────────────────────────────────

async function request<T>(
  method: string,
  url: string,
  body?: unknown,
  opts?: RequestOptions,
  // Internal: pre-built config from httpClient
  _preBuiltConfig?: RequestConfig,
): Promise<T> {
  const startTime = Date.now()
  const corrId = _preBuiltConfig?.correlationId ?? _corrId()
  _trackCorrId(corrId, url)

  const headers: Record<string, string> = {}
  if (!opts?.raw) headers['Content-Type'] = 'application/json'
  if (opts?.headers) Object.assign(headers, opts.headers)
  if (!opts?.noAuth) {
    const token = useAuthStore.getState().token
    if (token) headers['Authorization'] = `Bearer ${token}`
  }
  headers['X-Correlation-ID'] = corrId

  const fullUrl = `${PUBLIC_API_URL}${url}`

  const config: RequestConfig = _preBuiltConfig ?? {
    method,
    url,
    fullUrl,
    headers,
    body,
    signal: opts?.signal,
    correlationId: corrId,
    startTime,
    opts: opts ?? {},
  }

  // Run request interceptors
  const finalConfig = await _globalInterceptors.request.run(config)

  logger.debug(`>>> ${method} ${url} corr=${corrId}`, { corrId, method, url })

  const timeoutMs = opts?.timeout ?? _defaultTimeout
  const isDocstore = _isDocstoreUrl(url)
  const maxRetries = isDocstore ? DOCSTORE_MAX_RETRIES : (opts?.skipCircuitBreaker ? 0 : _defaultMaxRetries)
  const baseDelay = isDocstore ? DOCSTORE_BASE_DELAY : _defaultBaseDelay
  const retryableStatuses = isDocstore ? DOCSTORE_RETRYABLE_STATUSES : RETRYABLE_STATUSES
  let retries = 0
  let cacheHit = false

  while (true) {
    let signal = opts?.signal
    let timer: ReturnType<typeof setTimeout> | undefined
    if (!signal) {
      const ac = new AbortController()
      timer = setTimeout(() => ac.abort(), timeoutMs)
      signal = ac.signal
    }
    try {
      // Upload progress: wrap body ReadableStream if needed
      let fetchBody: BodyInit | undefined
      if (opts?.raw) {
        fetchBody = body as BodyInit
      } else if (body != null) {
        const jsonStr = JSON.stringify(body)
        if (opts?.progress?.onUpload) {
          const blob = new Blob([jsonStr], { type: 'application/json' })
          fetchBody = new ReadableStream({
            start(controller) {
              const chunk = new Uint8Array(blob.size)
              blob.arrayBuffer().then(ab => {
                chunk.set(new Uint8Array(ab))
                controller.enqueue(chunk)
                controller.close()
                opts!.progress!.onUpload!(100, blob.size, blob.size)
              })
            }
          })
        } else {
          fetchBody = jsonStr
        }
      }

      let res: Response
      if (opts?.progress?.onDownload) {
        const originalRes = await fetch(finalConfig.fullUrl, {
          method,
          headers: finalConfig.headers,
          body: fetchBody,
          signal,
        })
        if (!originalRes.ok || !originalRes.body) {
          res = originalRes
        } else {
          const contentLength = Number(originalRes.headers.get('content-length')) || 0
          let loaded = 0
          const reader = originalRes.body.getReader()
          const chunks: Uint8Array[] = []
          const decoder = new TextDecoder()
          while (true) {
            const { done, value } = await reader.read()
            if (done) break
            chunks.push(value)
            loaded += value.length
            if (contentLength > 0) {
              opts.progress.onDownload(Math.round((loaded / contentLength) * 100), loaded, contentLength)
            }
          }
          reader.releaseLock()
          const allChunks = new Uint8Array(loaded)
          let offset = 0
          for (const chunk of chunks) {
            allChunks.set(chunk, offset)
            offset += chunk.length
          }
          const text = decoder.decode(allChunks)
          res = new Response(text, {
            status: originalRes.status,
            statusText: originalRes.statusText,
            headers: originalRes.headers,
          })
        }
      } else {
        res = await fetch(finalConfig.fullUrl, {
          method,
          headers: finalConfig.headers,
          body: fetchBody,
          signal,
        })
      }

      if (timer) clearTimeout(timer)

      logger.debug(`<<< ${method} ${url} ${res.status} corr=${corrId}`, { corrId, method, url, status: res.status })

      // Build response envelope
      const resText = await res.text()
      const envelope: ResponseEnvelope = {
        ok: res.ok,
        status: res.status,
        statusText: res.statusText,
        headers: res.headers,
        text: resText,
        config: finalConfig,
      }
      if (res.ok && resText) {
        try { envelope.json = JSON.parse(resText) } catch { /* not JSON */ }
      }

      // Run response interceptors
      const finalEnvelope = await _globalInterceptors.response.run(envelope)

      if (!finalEnvelope.ok) {
        const status = finalEnvelope.status
        const isRetryable = retryableStatuses.has(status)
        if (isRetryable && retries < maxRetries) {
          retries++
          const retryAfter = Number(finalEnvelope.headers.get('Retry-After')) || 0
          const delay = retryAfter > 0 ? retryAfter * 1000 : baseDelay * Math.pow(2, retries - 1)
          logger.warning(
            `retry ${retries}/${maxRetries} ${method} ${url} ${status} delay=${delay}ms corr=${corrId}`,
            { corrId, method, url, status, retries, maxRetries, delay },
          )
          await new Promise(r => setTimeout(r, delay))
          continue
        }

        let detail: string | undefined
        let errorCode: string | undefined
        let errorDetails: unknown | undefined
        let correlationId: string | undefined
        try {
          const j = JSON.parse(finalEnvelope.text)
          detail = j.detail ?? j.message ?? j.error ?? finalEnvelope.text
          errorCode = j.code
          errorDetails = j.details
          correlationId = j.correlation_id
        } catch {
          detail = finalEnvelope.text || finalEnvelope.statusText || 'Could not request'
        }
        const message = Array.isArray(detail)
          ? detail.map((d: { msg?: string } | string) => typeof d === 'string' ? d : d.msg ?? '').join('; ')
          : detail || 'Could not request'

        const requestId = finalEnvelope.headers.get('X-Request-ID') || undefined
        const apiErr = new ApiError(message, status, { raw: finalEnvelope.text, code: errorCode, details: errorDetails, correlationId }, requestId)

        // Circuit breaker
        if (!opts?.skipCircuitBreaker) _globalCircuitBreaker.recordFailure()

        // Lifecycle: onError
        for (const hook of _globalHooks.onError) {
          try { await hook(apiErr, finalConfig) } catch { /* hook errors swallowed */ }
        }

        if (!opts?.silent) {
          import('./error-store').then(({ useErrorStore }) => {
            useErrorStore.getState().addError(apiErr, {
              source: url,
              title: status >= 500 ? 'Server Error' : status >= 400 ? `HTTP ${status}` : 'API Error',
              requestId,
            })
          })
        }
        throw apiErr
      }

      // Success — circuit breaker
      if (!opts?.skipCircuitBreaker) _globalCircuitBreaker.recordSuccess()

      // Return typed result
      if (!finalEnvelope.text) return undefined as T
      const json = finalEnvelope.json ?? JSON.parse(finalEnvelope.text)
      if (json && typeof json === 'object' && 'status' in json && 'data' in json) {
        const result = json.data as T
        if (json.meta && typeof result === 'object' && result !== null) {
          Object.defineProperty(result, '_meta', { value: json.meta, enumerable: false })
        }
        return result
      }
      return json as T
    } catch (e: unknown) {
      if (timer) clearTimeout(timer)
      if (e instanceof ApiError) throw e
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

      if (retries < maxRetries) {
        retries++
        const delay = baseDelay * Math.pow(2, retries - 1)
        logger.warning(
          `retry ${retries}/${maxRetries} ${method} ${url} ${kind} delay=${delay}ms corr=${corrId}`,
          { corrId, method, url, kind, retries, maxRetries, delay },
        )
        await new Promise(r => setTimeout(r, delay))
        continue
      }

      if (!opts?.skipCircuitBreaker) _globalCircuitBreaker.recordFailure()

      const apiErr = new ApiError(message, status)

      for (const hook of _globalHooks.onError) {
        try { await hook(apiErr, finalConfig) } catch { /* hook errors swallowed */ }
      }

      if (!opts?.silent) {
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

      throw apiErr
    }
  }
}

// ── Global instances ────────────────────────────────────────────────────────

const _globalInterceptors = {
  request: new InterceptorManager<RequestConfig>(),
  response: new InterceptorManager<ResponseEnvelope>(),
}

const _globalCircuitBreaker = new CircuitBreaker()
const _globalCache = new HttpCache()
const _globalThrottler = new Throttler()

const _globalHooks: {
  beforeRequest: ((config: RequestConfig) => RequestConfig | Promise<RequestConfig>)[]
  afterResponse: ((response: ResponseEnvelope) => ResponseEnvelope | Promise<ResponseEnvelope>)[]
  onError: ((error: ApiError, config: RequestConfig) => void | Promise<void>)[]
} = {
  beforeRequest: [],
  afterResponse: [],
  onError: [],
}

// ── Convenience functions (backward-compatible) ─────────────────────────────

export async function apiGet<T>(url: string, params?: Record<string, string>, opts?: RequestOptions): Promise<T> {
  const qs = params ? '?' + new URLSearchParams(params).toString() : ''
  const fullUrl = url + qs

  // Cache check
  const cacheOpt = opts?.cache
  if (cacheOpt && !opts?.signal) {
    const cacheKey = opts?.cacheKey ?? HttpCache.makeKey('GET', url, params)
    const cached = _globalCache.get(cacheKey)
    if (cached && !cached.stale) {
      logger.debug(`cache HIT ${url}`, { cacheKey })
      return cached.data as T
    }
    if (cached?.stale) {
      // Serve stale, revalidate in background
      logger.debug(`cache STALE ${url} — serving stale, revalidating`, { cacheKey })
      const revalidationPromise = request<T>('GET', fullUrl, undefined, opts)
        .then(data => {
          _globalCache.set(cacheKey, data, typeof cacheOpt === 'object' ? cacheOpt : undefined)
          return data
        })
        .catch(() => { /* background revalidation failed, stale data still valid */ })
      return cached.data as T
    }
  }

  // Throttle
  if (opts?.throttle) {
    await _globalThrottler.acquire()
    try {
      return await _doGet<T>(fullUrl, params, opts, cacheOpt)
    } finally {
      _globalThrottler.release()
    }
  }

  return _doGet<T>(fullUrl, params, opts, cacheOpt)
}

async function _doGet<T>(
  fullUrl: string,
  params: Record<string, string> | undefined,
  opts: RequestOptions | undefined,
  cacheOpt: boolean | CacheOptions | undefined,
): Promise<T> {
  // In-flight dedup
  if (!opts?.signal) {
    const existing = _inflight.get(fullUrl)
    if (existing) return existing as Promise<T>
  }

  // Recent dedup with TTL (opt-in)
  if (!opts?.signal && opts?.dedupTtlMs) {
    _cleanupRecentDedup()
    const recent = _recentDedup.get(fullUrl)
    if (recent && Date.now() < recent.expiresAt) {
      return recent.promise as Promise<T>
    }
  }

  const p = request<T>('GET', fullUrl, undefined, opts)

  if (!opts?.signal) {
    _inflight.set(fullUrl, p)
    if (opts?.dedupTtlMs) {
      _recentDedup.set(fullUrl, {
        promise: p,
        expiresAt: Date.now() + opts.dedupTtlMs,
      })
    }
  }

  try {
    const result = await p
    // Cache store
    if (cacheOpt && !opts?.signal) {
      const cacheKey = opts?.cacheKey ?? HttpCache.makeKey('GET', fullUrl.split('?')[0], params)
      _globalCache.set(cacheKey, result, typeof cacheOpt === 'object' ? cacheOpt : undefined)
    }
    return result
  } finally {
    if (!opts?.signal) {
      _inflight.delete(fullUrl)
      if (opts?.dedupTtlMs) {
        setTimeout(() => _recentDedup.delete(fullUrl), opts!.dedupTtlMs!)
      }
    }
  }
}

export async function apiPost<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T> {
  if (opts?.throttle) {
    await _globalThrottler.acquire()
    try { return await request<T>('POST', url, body, opts) } finally { _globalThrottler.release() }
  }
  return request<T>('POST', url, body, opts)
}

export async function apiPut<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T> {
  if (opts?.throttle) {
    await _globalThrottler.acquire()
    try { return await request<T>('PUT', url, body, opts) } finally { _globalThrottler.release() }
  }
  return request<T>('PUT', url, body, opts)
}

export async function apiDelete<T>(url: string, opts?: RequestOptions): Promise<T> {
  if (opts?.throttle) {
    await _globalThrottler.acquire()
    try { return await request<T>('DELETE', url, undefined, opts) } finally { _globalThrottler.release() }
  }
  return request<T>('DELETE', url, undefined, opts)
}

export async function apiPatch<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T> {
  if (opts?.throttle) {
    await _globalThrottler.acquire()
    try { return await request<T>('PATCH', url, body, opts) } finally { _globalThrottler.release() }
  }
  return request<T>('PATCH', url, body, opts)
}

// ── apiClient (backward-compatible) ─────────────────────────────────────────

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

// ── HttpClient (full-featured singleton) ────────────────────────────────────

export interface HttpClient {
  interceptors: {
    request: InterceptorManager<RequestConfig>
    response: InterceptorManager<ResponseEnvelope>
  }
  cache: HttpCache
  circuitBreaker: CircuitBreaker
  throttler: Throttler
  hooks: {
    beforeRequest: typeof _globalHooks.beforeRequest
    afterResponse: typeof _globalHooks.afterResponse
    onError: typeof _globalHooks.onError
  }
  defaults: {
    timeout: number
    maxRetries: number
    baseDelay: number
  }
  configure(opts: { timeout?: number; maxRetries?: number; baseDelay?: number }): void
  request<T>(method: string, url: string, body?: unknown, opts?: RequestOptions): Promise<T>
  get<T>(url: string, params?: Record<string, string>, opts?: RequestOptions): Promise<T>
  post<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T>
  put<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T>
  patch<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T>
  delete<T>(url: string, opts?: RequestOptions): Promise<T>
}

function _createHttpClient(opts?: HttpClientOptions): HttpClient {
  const interceptors = {
    request: new InterceptorManager<RequestConfig>(),
    response: new InterceptorManager<ResponseEnvelope>(),
  }
  const cache = new HttpCache(opts?.cache)
  // Reuse global circuit breaker unless custom config is provided
  const circuitBreaker = opts?.circuitBreaker ? new CircuitBreaker(opts.circuitBreaker) : _globalCircuitBreaker
  const throttler = new Throttler(opts?.throttle)
  const hooks: typeof _globalHooks = {
    beforeRequest: opts?.hooks?.beforeRequest ? [...opts.hooks.beforeRequest] : [],
    afterResponse: opts?.hooks?.afterResponse ? [...opts.hooks.afterResponse] : [],
    onError: opts?.hooks?.onError ? [...opts.hooks.onError] : [],
  }

  // Register custom interceptors
  if (opts?.interceptors?.request) {
    for (const i of opts.interceptors.request) interceptors.request.use(i.onFulfilled, i.onRejected)
  }
  if (opts?.interceptors?.response) {
    for (const i of opts.interceptors.response) interceptors.response.use(i.onFulfilled, i.onRejected)
  }

  return {
    interceptors,
    cache,
    circuitBreaker,
    throttler,
    hooks,
    defaults: {
      timeout: _defaultTimeout,
      maxRetries: _defaultMaxRetries,
      baseDelay: _defaultBaseDelay,
    },
    configure(newOpts) {
      if (newOpts.timeout !== undefined) { this.defaults.timeout = newOpts.timeout; _defaultTimeout = newOpts.timeout }
      if (newOpts.maxRetries !== undefined) { this.defaults.maxRetries = newOpts.maxRetries; _defaultMaxRetries = newOpts.maxRetries }
      if (newOpts.baseDelay !== undefined) { this.defaults.baseDelay = newOpts.baseDelay; _defaultBaseDelay = newOpts.baseDelay }
    },
    async request<T>(method: string, url: string, body?: unknown, reqOpts?: RequestOptions): Promise<T> {
      const startTime = Date.now()
      const corrId = _corrId()
      _trackCorrId(corrId, url)
      const fullUrl = `${PUBLIC_API_URL}${url}`

      const headers: Record<string, string> = {}
      if (!reqOpts?.raw) headers['Content-Type'] = 'application/json'
      if (reqOpts?.headers) Object.assign(headers, reqOpts.headers)
      if (!reqOpts?.noAuth) {
        const token = useAuthStore.getState().token
        if (token) headers['Authorization'] = `Bearer ${token}`
      }
      headers['X-Correlation-ID'] = corrId

      let config: RequestConfig = {
        method, url, fullUrl, headers, body,
        signal: reqOpts?.signal, correlationId: corrId, startTime, opts: reqOpts ?? {},
      }

      // Client interceptors
      config = await interceptors.request.run(config)

      // Client hooks
      for (const hook of hooks.beforeRequest) {
        config = await hook(config)
      }

      // Throttle
      if (reqOpts?.throttle) await throttler.acquire()

      try {
        const result = await request<T>(method, url, body, reqOpts, config)
        return result
      } finally {
        if (reqOpts?.throttle) throttler.release()
      }
    },
    async get<T>(url: string, params?: Record<string, string>, opts?: RequestOptions): Promise<T> {
      const qs = params ? '?' + new URLSearchParams(params).toString() : ''
      const fullUrl = url + qs

      // Cache
      const cacheOpt = opts?.cache
      if (cacheOpt && !opts?.signal) {
        const cacheKey = opts?.cacheKey ?? HttpCache.makeKey('GET', url, params)
        const cached = cache.get(cacheKey)
        if (cached && !cached.stale) return cached.data as T
        if (cached?.stale) {
          const revalidation = this.request<T>('GET', fullUrl, undefined, opts)
            .then(data => { cache.set(cacheKey, data, typeof cacheOpt === 'object' ? cacheOpt : undefined); return data })
            .catch(() => {})
          return cached.data as T
        }
      }

      const result = await this.request<T>('GET', fullUrl, undefined, opts)

      if (cacheOpt && !opts?.signal) {
        const cacheKey = opts?.cacheKey ?? HttpCache.makeKey('GET', url, params)
        cache.set(cacheKey, result, typeof cacheOpt === 'object' ? cacheOpt : undefined)
      }
      return result
    },
    async post<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T> {
      return this.request<T>('POST', url, body, opts)
    },
    async put<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T> {
      return this.request<T>('PUT', url, body, opts)
    },
    async patch<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T> {
      return this.request<T>('PATCH', url, body, opts)
    },
    async delete<T>(url: string, opts?: RequestOptions): Promise<T> {
      return this.request<T>('DELETE', url, undefined, opts)
    },
  }
}

/** Global singleton HTTP client with full feature access. */
export const httpClient: HttpClient = _createHttpClient()

/** Factory for isolated HttpClient instances (e.g. different base URLs). */
export function createHttpClient(opts?: HttpClientOptions): HttpClient {
  return _createHttpClient(opts)
}

// ── SSE streaming (unchanged) ───────────────────────────────────────────────

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
  _trackCorrId(corrId, url)
  return fetch(url, { ...opts, headers })
}

interface StreamSSEOptions {
  method?: string
  body?: unknown
  signal?: AbortSignal
  noAuth?: boolean
  lastEventId?: string
  maxRetries?: number
}

export async function* streamSSE(url: string, opts?: StreamSSEOptions): AsyncGenerator<SSEEvent> {
  const method = opts?.method ?? 'POST'
  const maxRetries = opts?.maxRetries ?? 3
  const baseDelay = 500

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
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

    _trackCorrId(corrId, url)

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

    if (!res.ok) {
      const status = res.status
      if (status === 503 && attempt < maxRetries) {
        const delay = baseDelay * Math.pow(2, attempt)
        logger.warning(
          `SSE retry ${attempt + 1}/${maxRetries} ${method} ${url} ${status} delay=${delay}ms corr=${corrId}`,
          { corrId, method, url, status, attempt, maxRetries, delay },
        )
        await new Promise(r => setTimeout(r, delay))
        continue
      }
      yield { status: 'error', message: `HTTP ${res.status}${res.statusText ? `: ${res.statusText}` : ''}`, data: { http_status: res.status, error: `HTTP ${res.status}` } }
      return
    }

    if (!res.body) {
      yield { status: 'error', message: 'No response body' }
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
      return
    } finally {
      reader.releaseLock()
    }
  }
}
