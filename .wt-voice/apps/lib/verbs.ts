import { logger } from '../web/lib/dev-log'
import { HttpCache } from './resilience'
import type { CacheOptions, RequestOptions } from './types'
import { request } from './request'
import { _globalCache, _globalThrottler, _inflight, _recentDedup, _cleanupRecentDedup } from './state'

export async function apiGet<T>(
  url: string,
  params?: Record<string, string>,
  opts?: RequestOptions,
): Promise<T> {
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
      request<T>('GET', fullUrl, undefined, opts)
        .then((data) => {
          _globalCache.set(cacheKey, data, typeof cacheOpt === 'object' ? cacheOpt : undefined)
          return data
        })
        .catch(() => {
          /* background revalidation failed, stale data still valid */
        })
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
    try {
      return await request<T>('POST', url, body, opts)
    } finally {
      _globalThrottler.release()
    }
  }
  return request<T>('POST', url, body, opts)
}

/**
 * Multipart upload via FormData. Uses `raw` mode so the JSON Content-Type
 * is omitted — the browser sets `multipart/form-data` with its own boundary.
 */
export async function apiPostForm<T>(
  url: string,
  form: FormData,
  opts?: RequestOptions,
): Promise<T> {
  const rawOpts: RequestOptions = { ...opts, raw: true }
  if (rawOpts.throttle) {
    await _globalThrottler.acquire()
    try {
      return await request<T>('POST', url, form, rawOpts)
    } finally {
      _globalThrottler.release()
    }
  }
  return request<T>('POST', url, form, rawOpts)
}

export async function apiPut<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T> {
  if (opts?.throttle) {
    await _globalThrottler.acquire()
    try {
      return await request<T>('PUT', url, body, opts)
    } finally {
      _globalThrottler.release()
    }
  }
  return request<T>('PUT', url, body, opts)
}

export async function apiDelete<T>(url: string, opts?: RequestOptions): Promise<T> {
  if (opts?.throttle) {
    await _globalThrottler.acquire()
    try {
      return await request<T>('DELETE', url, undefined, opts)
    } finally {
      _globalThrottler.release()
    }
  }
  return request<T>('DELETE', url, undefined, opts)
}

export async function apiPatch<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T> {
  if (opts?.throttle) {
    await _globalThrottler.acquire()
    try {
      return await request<T>('PATCH', url, body, opts)
    } finally {
      _globalThrottler.release()
    }
  }
  return request<T>('PATCH', url, body, opts)
}
