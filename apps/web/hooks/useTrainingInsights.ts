'use client'

import { useState, useEffect, useCallback } from 'react'
import { settingsController, type AdaptiveInsights } from '@/lib/settings-controller'

export interface TrendPoint {
  name: string
  quality: number
  loss: number | null
  model: string
}

export interface UseTrainingInsightsReturn {
  insights: AdaptiveInsights | null
  trendPoints: TrendPoint[]
  loading: boolean
  useConfigStatus: 'idle' | 'applied'
  loadData: () => Promise<void>
  applyRecommendedConfig: () => Promise<void>
}

function toChartPoints(runs: Array<Record<string, unknown>> | undefined): TrendPoint[] {
  return (runs ?? []).map((r, i) => {
    const ts = (r.timestamp as number) ?? 0
    const date = ts ? new Date(ts * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : ''
    const loss = (r.final_loss as number) ?? 0
    return {
      name: `#${i + 1}`,
      date,
      quality: (r.quality_score as number) ?? 0,
      loss: loss > 0 ? loss : null,
      model: (r.model as string) ?? '',
    }
  })
}

export function useTrainingInsights(): UseTrainingInsightsReturn {
  const [insights, setInsights] = useState<AdaptiveInsights | null>(null)
  const [trendPoints, setTrendPoints] = useState<TrendPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [useConfigStatus, setUseConfigStatus] = useState<'idle' | 'applied'>('idle')

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const [insightsResp, trendResp] = await Promise.all([
        settingsController.getAdaptiveInsights().catch(() => null),
        settingsController.exportTrainingHistory('json', 200).catch(() => null),
      ])
      setInsights(insightsResp)
      setTrendPoints(toChartPoints(trendResp?.outcomes))
    } catch (err) {
      console.error('Failed to load insights:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void loadData() }, [loadData])

  const applyRecommendedConfig = useCallback(async () => {
    if (!insights?.best_config) return
    await settingsController.updateTraining({
      preferred_model: (insights.best_config.model as unknown as string) || 'gpt2',
      preferred_method: 'finetune',
      max_checkpoints: 10,
    })
    setUseConfigStatus('applied')
    setTimeout(() => setUseConfigStatus('idle'), 3000)
  }, [insights])

  return { insights, trendPoints, loading, useConfigStatus, loadData, applyRecommendedConfig }
}
