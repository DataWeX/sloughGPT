import type { ApiError } from './errors'

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
  /** Cold-start grace period (ms) during which failures are not counted. Default 15000. */
  gracePeriodMs?: number
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
