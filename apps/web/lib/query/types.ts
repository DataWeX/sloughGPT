/** Query system types — composable, generic request management for the web UI. */

import type { DependencyList } from 'react'

/** Serializable query key — string or array (serialized to cache key). */
export type QueryKey = string | readonly unknown[]

export type QueryStatus = 'idle' | 'loading' | 'success' | 'error'

export interface QueryState<T = unknown> {
  data: T | undefined
  error: Error | null
  status: QueryStatus
  isFetching: boolean
  updatedAt: number
}

export interface QueryOptions<T = unknown> {
  /** Skip auto-fetch (manual refetch only). Default: true */
  enabled?: boolean
  /** Data considered fresh for this many ms. Default: 0 (always stale). */
  staleTime?: number
  /** Keep unused data in cache for this many ms. Default: 5 min. */
  gcTime?: number
  /** Retry count on error. Default: 0 */
  retry?: number
  /** Base delay between retries (ms). Default: 1000 */
  retryDelay?: number
  /** Refetch when component mounts. Default: true */
  refetchOnMount?: boolean
  /** Called with data on successful fetch. */
  onSuccess?: (data: T) => void
  /** Called with error on failed fetch. */
  onError?: (err: Error) => void
}

export interface MutationOptions<T, V = void> {
  onSuccess?: (data: T, vars: V) => void
  onError?: (err: Error, vars: V) => void
  onSettled?: (data: T | undefined, err: Error | null, vars: V) => void
  /** Query keys to invalidate after successful mutation. */
  invalidateKeys?: QueryKey[]
}

export interface QueryResult<T> {
  data: T | undefined
  error: Error | null
  isLoading: boolean
  isFetching: boolean
  isStale: boolean
  status: QueryStatus
  refetch: () => Promise<T>
}

export interface MutationResult<T, V> {
  mutate: (vars: V) => void
  mutateAsync: (vars: V) => Promise<T>
  isLoading: boolean
  error: Error | null
  reset: () => void
}

/** Serialize a query key to string for cache lookup. */
export function serializeKey(key: QueryKey): string {
  if (typeof key === 'string') return key
  return JSON.stringify(key)
}
