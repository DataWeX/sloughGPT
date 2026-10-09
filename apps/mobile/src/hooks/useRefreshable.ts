import { useState, useCallback, useEffect, useRef } from 'react'

/**
 * Shared loading/refreshing hook for mobile screens.
 * Manages loading state, refreshing state, and initial fetch.
 *
 * Usage:
 *   const { loading, refreshing, onRefresh } = useRefreshable(fetchData);
 */
export function useRefreshable(fetchFn: () => Promise<void>) {
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    fetchFn().finally(() => {
      if (mountedRef.current) setLoading(false)
    })
    return () => { mountedRef.current = false }
  }, [fetchFn])

  const onRefresh = useCallback(async () => {
    setRefreshing(true)
    try {
      await fetchFn()
    } finally {
      if (mountedRef.current) setRefreshing(false)
    }
  }, [fetchFn])

  return { loading, refreshing, onRefresh }
}
