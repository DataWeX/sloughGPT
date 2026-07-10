/** React hooks for the query system.

  Usage:
    // Generic data fetching
    const { data: models, isLoading } = useQuery('models', () => api.getModels())

    // With params key — re-fetches when key changes
    const { data } = useQuery(['checkpoints', soulName], () => api.listCheckpoints(soulName))

    // Mutations with auto-invalidation
    const { mutate: deleteModel, isLoading: deleting } = useMutation(
      (id: string) => api.deleteModel(id),
      { invalidateKeys: ['models'] }
    )

    // Manual cache control
    const invalidate = useInvalidate()
    invalidate('models')
*/

'use client'

import { useEffect, useCallback, useRef, useState, useSyncExternalStore } from 'react'
import {
  serializeKey,
  type QueryKey,
  type QueryOptions,
  type QueryResult,
  type MutationOptions,
  type MutationResult,
  type QueryState,
} from './types'
import {
  useQueryStore,
  fetchQuery,
  invalidateQuery,
  subscribeQuery,
  isStale as checkStale,
} from './client'

// ─── useQuery ───────────────────────────────────────────────────────────────

export function useQuery<T>(
  key: QueryKey,
  fetcher: () => Promise<T>,
  options: QueryOptions<T> = {},
): QueryResult<T> {
  const {
    enabled = true,
    staleTime,
    gcTime,
    retry,
    retryDelay,
    refetchOnMount = true,
    onSuccess,
    onError,
  } = options

  const sk = serializeKey(key)
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher
  const optsRef = useRef({ staleTime, gcTime, retry, retryDelay })
  optsRef.current = { staleTime, gcTime, retry, retryDelay }
  const callbacksRef = useRef({ onSuccess, onError })
  callbacksRef.current = { onSuccess, onError }

  // Subscribe to store changes via useSyncExternalStore
  const subscribe = useCallback(
    (cb: () => void) => {
      const unsub = subscribeQuery(key)
      // Also subscribe to store changes
      const unsubStore = useQueryStore.subscribe(cb)
      return () => { unsub(); unsubStore() }
    },
    [key],
  )

  const snapRef = useRef<QueryState<T>>({
    data: undefined, error: null, status: 'idle', isFetching: false, updatedAt: 0,
  })
  const getSnapshot = useCallback((): QueryState<T> => {
    const entry = useQueryStore.getState().cache[sk]
    if (!entry) {
      if (snapRef.current.status !== 'idle') {
        snapRef.current = { data: undefined, error: null, status: 'idle', isFetching: false, updatedAt: 0 }
      }
      return snapRef.current
    }
    const s = snapRef.current
    if (s.data === (entry.data as T) && s.error === entry.error && s.status === entry.status && s.isFetching === entry.isFetching && s.updatedAt === entry.updatedAt) {
      return s
    }
    snapRef.current = { data: entry.data as T, error: entry.error, status: entry.status, isFetching: entry.isFetching, updatedAt: entry.updatedAt }
    return snapRef.current
  }, [sk])

  const state = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)

  // Trigger fetch on mount / key change
  useEffect(() => {
    if (!enabled) return

    const doFetch = refetchOnMount && (state.data === undefined || checkStale(key))
    if (!doFetch) return

    let cancelled = false
    ;(async () => {
      try {
        const data = await fetchQuery(key, () => fetcherRef.current(), optsRef.current)
        if (!cancelled) {
          callbacksRef.current.onSuccess?.(data)
        }
      } catch (err) {
        if (!cancelled) {
          callbacksRef.current.onError?.(err instanceof Error ? err : new Error(String(err)))
        }
      }
    })()

    return () => { cancelled = true }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sk, enabled])

  const refetch = useCallback(async (): Promise<T> => {
    return fetchQuery(key, () => fetcherRef.current(), optsRef.current)
  }, [key])

  const isLoading = state.status === 'loading' && state.data === undefined
  const isStale_ = state.status === 'success' && checkStale(key)

  return {
    data: state.data as T | undefined,
    error: state.error,
    isLoading,
    isFetching: state.isFetching,
    isStale: isStale_,
    status: state.status,
    refetch,
  }
}

// ─── useMutation ────────────────────────────────────────────────────────────

export function useMutation<T, V = void>(
  fn: (vars: V) => Promise<T>,
  options: MutationOptions<T, V> = {},
): MutationResult<T, V> {
  const fnRef = useRef(fn)
  fnRef.current = fn
  const callbacksRef = useRef({ onSuccess: options.onSuccess, onError: options.onError, onSettled: options.onSettled })
  callbacksRef.current = { onSuccess: options.onSuccess, onError: options.onError, onSettled: options.onSettled }
  const invalidateKeysRef = useRef(options.invalidateKeys)
  invalidateKeysRef.current = options.invalidateKeys
  const [state, setState] = useState<{ isLoading: boolean; error: Error | null }>({
    isLoading: false,
    error: null,
  })

  const mutateAsync = useCallback(
    async (vars: V): Promise<T> => {
      setState({ isLoading: true, error: null })
      try {
        const data = await fnRef.current(vars)
        setState({ isLoading: false, error: null })
        const cb = callbacksRef.current
        cb.onSuccess?.(data, vars)
        cb.onSettled?.(data, null, vars)
        const keys = invalidateKeysRef.current
        if (keys) {
          for (const k of keys) {
            invalidateQuery(k)
          }
        }
        return data
      } catch (err) {
        const e = err instanceof Error ? err : new Error(String(err))
        setState({ isLoading: false, error: e })
        const cb = callbacksRef.current
        cb.onError?.(e, vars)
        cb.onSettled?.(undefined, e, vars)
        throw e
      }
    },
    [],
  )

  const mutate = useCallback((vars: V) => { mutateAsync(vars).catch(() => {}) }, [mutateAsync])

  const reset = useCallback(() => {
    setState({ isLoading: false, error: null })
  }, [])

  return { mutate, mutateAsync, isLoading: state.isLoading, error: state.error, reset }
}

// ─── useInvalidate ──────────────────────────────────────────────────────────

export function useInvalidate(): (key: QueryKey) => void {
  return useCallback((key: QueryKey) => { invalidateQuery(key) }, [])
}

// ─── useIsFetching ──────────────────────────────────────────────────────────

export function useIsFetching(): number {
  const subscribe = useCallback(
    (cb: () => void) => useQueryStore.subscribe(cb),
    [],
  )
  const getSnapshot = useCallback(
    () => useQueryStore.getState().fetchingKeys.size,
    [],
  )
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
}
