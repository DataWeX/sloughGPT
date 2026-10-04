import { ApiError } from './errors'
import type { CacheOptions, CircuitBreakerOptions, CircuitState, ThrottleOptions } from './types'

const DEFAULT_CACHE_TTL_MS = 60_000
const DEFAULT_CB_FAILURE_THRESHOLD = 5
const DEFAULT_CB_RESET_TIMEOUT_MS = 30_000
const DEFAULT_CB_GRACE_MS = 15_000
const DEFAULT_THROTTLE_MAX = 10
const DEFAULT_THROTTLE_QUEUE = 50
const DEFAULT_THROTTLE_TIMEOUT_MS = 10_000

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

export class CircuitBreaker {
  private _state: CircuitState = 'closed'
  private _failureCount = 0
  private _successCount = 0
  private _lastFailureAt = 0
  private _halfOpenAttempts = 0
  private _createdAt = Date.now()

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
    // During the first GRACE_MS after creation (cold start grace period),
    // don't count failures toward the circuit breaker threshold.
    // The backend may still be loading modules / models.
    const graceMs = this.opts.gracePeriodMs ?? DEFAULT_CB_GRACE_MS
    if (Date.now() - this._createdAt < graceMs) return

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

  get failureCount() {
    return this._failureCount
  }
  get successCount() {
    return this._successCount
  }
}

interface ThrottleQueueEntry {
  resolve: () => void
  reject: (err: Error) => void
  timer?: ReturnType<typeof setTimeout>
}

export class Throttler {
  private _running = 0
  private _queue: ThrottleQueueEntry[] = []

  constructor(private opts: Partial<ThrottleOptions> = {}) {}

  get running() {
    return this._running
  }
  get queued() {
    return this._queue.length
  }

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
        const idx = this._queue.findIndex((e) => e.resolve === resolve)
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
