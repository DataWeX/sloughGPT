import { useAuthStore } from '../auth'
import { InterceptorManager } from './interceptors'
import { HttpCache, CircuitBreaker, Throttler } from './resilience'
import type { HttpClientOptions, RequestConfig, ResponseEnvelope, RequestOptions } from './types'
import { _corrId, _trackCorrId } from './corr-id'
import { _resolveUrl } from './url'
import { httpDefaults } from './retry'
import { _globalCircuitBreaker, _globalHooks } from './state'
import { request } from './request'

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
  const circuitBreaker = opts?.circuitBreaker
    ? new CircuitBreaker(opts.circuitBreaker)
    : _globalCircuitBreaker
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
    for (const i of opts.interceptors.response)
      interceptors.response.use(i.onFulfilled, i.onRejected)
  }

  return {
    interceptors,
    cache,
    circuitBreaker,
    throttler,
    hooks,
    defaults: {
      timeout: httpDefaults.timeout,
      maxRetries: httpDefaults.maxRetries,
      baseDelay: httpDefaults.baseDelay,
    },
    configure(newOpts) {
      if (newOpts.timeout !== undefined) {
        this.defaults.timeout = newOpts.timeout
        httpDefaults.timeout = newOpts.timeout
      }
      if (newOpts.maxRetries !== undefined) {
        this.defaults.maxRetries = newOpts.maxRetries
        httpDefaults.maxRetries = newOpts.maxRetries
      }
      if (newOpts.baseDelay !== undefined) {
        this.defaults.baseDelay = newOpts.baseDelay
        httpDefaults.baseDelay = newOpts.baseDelay
      }
    },
    async request<T>(
      method: string,
      url: string,
      body?: unknown,
      reqOpts?: RequestOptions,
    ): Promise<T> {
      const startTime = Date.now()
      const corrId = _corrId()
      _trackCorrId(corrId, url)
      const fullUrl = _resolveUrl(url)

      const headers: Record<string, string> = {}
      if (!reqOpts?.raw) headers['Content-Type'] = 'application/json'
      if (reqOpts?.headers) Object.assign(headers, reqOpts.headers)
      if (!reqOpts?.noAuth) {
        const token = useAuthStore.getState().token
        if (token) headers['Authorization'] = `Bearer ${token}`
      }
      headers['X-Correlation-ID'] = corrId

      let config: RequestConfig = {
        method,
        url,
        fullUrl,
        headers,
        body,
        signal: reqOpts?.signal,
        correlationId: corrId,
        startTime,
        opts: reqOpts ?? {},
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
          this.request<T>('GET', fullUrl, undefined, opts)
            .then((data) => {
              cache.set(cacheKey, data, typeof cacheOpt === 'object' ? cacheOpt : undefined)
              return data
            })
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
