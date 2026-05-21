/** QueryClient — global cache and fetch management.

  Provides:
  - Deduplication of in-flight requests (same key = same promise)
  - Configurable stale time and garbage collection
  - Manual invalidation
  - Subscriber tracking for cache cleanup
*/

'use client'

import { create } from 'zustand'
import { serializeKey, type QueryKey, type QueryState, type QueryStatus } from './types'

const DEFAULT_STALE_TIME = 0
const DEFAULT_GC_TIME = 5 * 60 * 1000
const DEFAULT_RETRY = 0
const DEFAULT_RETRY_DELAY = 1000

interface QueryCacheEntry<T = unknown> {
  data: T | undefined
  error: Error | null
  status: QueryStatus
  isFetching: boolean
  updatedAt: number
  _promise?: Promise<T>
  _subscribers: number
  _staleTime: number
  _gcTime: number
  _gcTimer?: ReturnType<typeof setTimeout>
}

interface QueryStoreState {
  cache: Record<string, QueryCacheEntry>
  /** Set of all currently fetching keys (for useIsFetching). */
  fetchingKeys: Set<string>
}

export const useQueryStore = create<QueryStoreState>(() => ({
  cache: {},
  fetchingKeys: new Set(),
}))

function getEntry(key: string): QueryCacheEntry | undefined {
  return useQueryStore.getState().cache[key]
}

function setEntry(key: string, partial: Partial<QueryCacheEntry>) {
  useQueryStore.setState(state => {
    const existing = state.cache[key] || {
      data: undefined,
      error: null,
      status: 'idle' as QueryStatus,
      isFetching: false,
      updatedAt: 0,
      _subscribers: 0,
      _staleTime: DEFAULT_STALE_TIME,
      _gcTime: DEFAULT_GC_TIME,
    }
    state.cache[key] = { ...existing, ...partial }
    return { cache: { ...state.cache } }
  })
}

function scheduleGC(key: string, gcTime: number) {
  const entry = getEntry(key)
  if (!entry) return
  if (entry._gcTimer) clearTimeout(entry._gcTimer)
  if (entry._subscribers > 0) return // still in use
  const timer = setTimeout(() => {
    useQueryStore.setState(state => {
      const next = { ...state.cache }
      delete next[key]
      return { cache: next }
    })
  }, gcTime)
  setEntry(key, { _gcTimer: timer })
}

function cancelGC(key: string) {
  const entry = getEntry(key)
  if (entry?._gcTimer) {
    clearTimeout(entry._gcTimer)
    setEntry(key, { _gcTimer: undefined })
  }
}

export interface FetchOptions {
  staleTime?: number
  gcTime?: number
  retry?: number
  retryDelay?: number
}

async function executeFetch<T>(
  key: string,
  fetcher: () => Promise<T>,
  options: FetchOptions = {}
): Promise<T> {
  const { retry = DEFAULT_RETRY, retryDelay = DEFAULT_RETRY_DELAY } = options

  let lastError: Error | null = null
  for (let attempt = 0; attempt <= retry; attempt++) {
    try {
      return await fetcher()
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err))
      if (attempt < retry) {
        await new Promise(r => setTimeout(r, retryDelay * (attempt + 1)))
      }
    }
  }
  throw lastError!
}

/** Fetch or retrieve cached data for a key. Returns data and whether it was fresh. */
export async function fetchQuery<T>(
  key: QueryKey,
  fetcher: () => Promise<T>,
  options: FetchOptions = {}
): Promise<T> {
  const sk = serializeKey(key)
  const entry = getEntry(sk)

  // Return cached fresh data
  if (entry && entry.status === 'success' && entry.data !== undefined) {
    const staleTime = options.staleTime ?? entry._staleTime ?? DEFAULT_STALE_TIME
    if (Date.now() - entry.updatedAt < staleTime) {
      return entry.data as T
    }
  }

  // Dedup in-flight requests
  if (entry?._promise) {
    return entry._promise as Promise<T>
  }

  // Mark fetching
  useQueryStore.setState(state => {
    state.fetchingKeys.add(sk)
    return { fetchingKeys: new Set(state.fetchingKeys) }
  })
  setEntry(sk, {
    status: 'loading',
    isFetching: true,
    error: null,
    _staleTime: options.staleTime ?? DEFAULT_STALE_TIME,
    _gcTime: options.gcTime ?? DEFAULT_GC_TIME,
  })

  const promise = executeFetch(sk, fetcher, options)
    .then(data => {
      setEntry(sk, {
        data,
        status: 'success',
        isFetching: false,
        error: null,
        updatedAt: Date.now(),
        _promise: undefined,
      })
      useQueryStore.setState(state => {
        state.fetchingKeys.delete(sk)
        return { fetchingKeys: new Set(state.fetchingKeys) }
      })
      return data
    })
    .catch((err: Error) => {
      setEntry(sk, {
        status: 'error',
        isFetching: false,
        error: err,
        _promise: undefined,
      })
      useQueryStore.setState(state => {
        state.fetchingKeys.delete(sk)
        return { fetchingKeys: new Set(state.fetchingKeys) }
      })
      throw err
    })

  setEntry(sk, { _promise: promise })
  return promise
}

/** Invalidate a cache entry — triggers refetch on next subscriber mount. */
export function invalidateQuery(key: QueryKey) {
  const sk = serializeKey(key)
  const entry = getEntry(sk)
  if (entry) {
    setEntry(sk, { updatedAt: 0, _promise: undefined })
  }
}

/** Subscribe a component to a query key. Returns unsubscribe function. */
export function subscribeQuery(key: QueryKey): () => void {
  const sk = serializeKey(key)
  cancelGC(sk)
  const entry = getEntry(sk)
  setEntry(sk, { _subscribers: (entry?._subscribers ?? 0) + 1 })
  return () => {
    const e = getEntry(sk)
    if (!e) return
    const next = e._subscribers - 1
    setEntry(sk, { _subscribers: next })
    if (next <= 0) {
      scheduleGC(sk, e._gcTime ?? DEFAULT_GC_TIME)
    }
  }
}

/** Get the current state for a key (for use outside React). */
export function getQueryState<T>(key: QueryKey): QueryState<T> {
  const sk = serializeKey(key)
  const entry = getEntry(sk)
  if (!entry) {
    return { data: undefined, error: null, status: 'idle', isFetching: false, updatedAt: 0 }
  }
  return { data: entry.data as T, error: entry.error, status: entry.status, isFetching: entry.isFetching, updatedAt: entry.updatedAt }
}

export function isStale(key: QueryKey): boolean {
  const sk = serializeKey(key)
  const entry = getEntry(sk)
  if (!entry) return true
  return Date.now() - entry.updatedAt >= (entry._staleTime ?? DEFAULT_STALE_TIME)
}
