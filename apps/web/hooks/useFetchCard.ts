'use client'

import { useState, useEffect, useCallback } from 'react'

/**
 * Generic hook for card components that need fetch + loading + error + retry.
 * Handles active-flag cleanup to avoid setting state on unmounted components.
 *
 * @param fetchFn - Async function that returns the data
 * @param deps - Dependencies that trigger a re-fetch (typically the controller)
 * @returns { data, loading, error, refetch }
 */
export function useFetchCard<T>(
  fetchFn: () => Promise<T>,
  deps: readonly unknown[] = [],
) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refetch = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await fetchFn()
      setData(result)
    } catch {
      setError('Failed to load')
    } finally {
      setLoading(false)
    }
  }, deps)

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      setError(null)
      try {
        const result = await fetchFn()
        if (active) setData(result)
      } catch {
        if (active) setError('Failed to load')
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, deps)

  return { data, loading, error, refetch } as const
}
