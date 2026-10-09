'use client'

import { useEffect, useState, useCallback } from 'react'
import { apiGet } from '@/lib/http-client'

const POLL_INTERVAL = 30_000

export function useConsciousnessStats() {
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStats = useCallback(async () => {
    try {
      const data = await apiGet('/consciousness/stats')
      setStats(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch stats')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await apiGet('/consciousness/stats')
        if (!active) return
        setStats(data)
        setError(null)
      } catch (err) {
        if (!active) return
        setError(err instanceof Error ? err.message : 'Failed to fetch stats')
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    const id = setInterval(fetchStats, POLL_INTERVAL)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [fetchStats])

  return { stats, loading, error, refetch: fetchStats }
}
