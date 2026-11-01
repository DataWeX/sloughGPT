import { InterceptorManager } from './interceptors'
import { HttpCache, CircuitBreaker, Throttler } from './resilience'
import type { ApiError } from './errors'
import type { RequestConfig, ResponseEnvelope } from './types'

export const _inflight = new Map<string, Promise<unknown>>()
export const _recentDedup = new Map<string, { promise: Promise<unknown>; expiresAt: number }>()

export function _cleanupRecentDedup() {
  const now = Date.now()
  for (const [key, entry] of _recentDedup) {
    if (now > entry.expiresAt) _recentDedup.delete(key)
  }
}

export const _globalInterceptors = {
  request: new InterceptorManager<RequestConfig>(),
  response: new InterceptorManager<ResponseEnvelope>(),
}

export const _globalCircuitBreaker = new CircuitBreaker()
export const _globalCache = new HttpCache()
export const _globalThrottler = new Throttler()

export const _globalHooks: {
  beforeRequest: ((config: RequestConfig) => RequestConfig | Promise<RequestConfig>)[]
  afterResponse: ((response: ResponseEnvelope) => ResponseEnvelope | Promise<ResponseEnvelope>)[]
  onError: ((error: ApiError, config: RequestConfig) => void | Promise<void>)[]
} = {
  beforeRequest: [],
  afterResponse: [],
  onError: [],
}
