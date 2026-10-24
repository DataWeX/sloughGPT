'use client'

import { useState, useEffect, useCallback } from 'react'
import { settingsController } from '@/lib/settings-controller'

export interface TrainingAnalytics {
  total_runs: number
  avg_quality: number
  convergence_rate: number
  quality_trend: Array<{ run_id: string; quality: number; timestamp: string }>
  method_distribution: Record<string, number>
  model_distribution: Record<string, number>
  tag_cloud: Record<string, number>
  best_run: { run_id: string; model: string; quality_score: number; method: string } | null
  recent_runs: Array<{
    run_id: string; model: string; method: string; quality_score: number
    converged: boolean; timestamp: string; tags: string[]
  }>
}

export interface UseTrainingAnalyticsReturn {
  analytics: TrainingAnalytics | null
  loading: boolean
  error: string | null
  load: () => Promise<void>
}

export function useTrainingAnalytics(): UseTrainingAnalyticsReturn {
  const [analytics, setAnalytics] = useState<TrainingAnalytics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await settingsController.getTrainingAnalytics()
      setAnalytics(data as unknown as TrainingAnalytics)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load analytics')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void load() }, [load])

  return { analytics, loading, error, load }
}
