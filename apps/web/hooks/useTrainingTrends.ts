'use client'

import { useState, useEffect, useCallback } from 'react'
import { apiGet } from '@/lib/http-client'

export interface TrendRunSummary {
  run_id: string
  timestamp: number
  model: string
  dataset: string
  method: string
  quality_score: number
  final_loss: number
  perplexity: number
  converged: boolean
  early_stopped: boolean
}

export interface TrendModelSummary {
  model: string
  total_runs: number
  avg_quality: number
  best_quality: number
  avg_loss: number
  latest_quality: number
  improving: boolean
}

export interface TrendSummary {
  total_runs: number
  avg_quality: number
  best_quality: number
  avg_loss: number
  trend: 'improving' | 'declining' | 'stable'
}

export interface TrendChartPoint {
  name: string
  date: string
  quality: number
  loss: number | null
  perplexity: number | null
  dataset: string
  model: string
  method: string
}

export interface UseTrainingTrendsReturn {
  runs: TrendRunSummary[]
  models: TrendModelSummary[]
  summary: TrendSummary | null
  chartPoints: TrendChartPoint[]
  loading: boolean
  modelFilter: string
  datasetFilter: string
  setModelFilter: (v: string) => void
  setDatasetFilter: (v: string) => void
  loadTrends: () => Promise<void>
}

function loadTrendsData(modelFilter: string, datasetFilter: string) {
  const params = new URLSearchParams()
  if (modelFilter) params.set('model', modelFilter)
  if (datasetFilter) params.set('dataset', datasetFilter)
  const qs = params.toString()
  return apiGet<{ runs: TrendRunSummary[]; models: TrendModelSummary[]; summary: TrendSummary }>(
    `/training/trends${qs ? `?${qs}` : ''}`,
  )
}

export function useTrainingTrends(): UseTrainingTrendsReturn {
  const [runs, setRuns] = useState<TrendRunSummary[]>([])
  const [models, setModels] = useState<TrendModelSummary[]>([])
  const [summary, setSummary] = useState<TrendSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [modelFilter, setModelFilter] = useState('')
  const [datasetFilter, setDatasetFilter] = useState('')

  const loadTrends = useCallback(async () => {
    setLoading(true)
    try {
      const resp = await loadTrendsData(modelFilter, datasetFilter)
      setRuns(resp.runs ?? [])
      setModels(resp.models ?? [])
      setSummary(resp.summary ?? null)
    } catch (err) {
      console.error('Failed to load trends:', err)
    } finally {
      setLoading(false)
    }
  }, [modelFilter, datasetFilter])

  useEffect(() => { void loadTrends() }, [loadTrends])

  const chartPoints: TrendChartPoint[] = runs.map(r => ({
    name: `${r.model || 'unknown'}#${r.run_id.slice(-6)}`,
    date: r.timestamp ? new Date(r.timestamp * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : '—',
    quality: r.quality_score,
    loss: r.final_loss > 0 ? r.final_loss : null,
    perplexity: r.perplexity > 0 ? r.perplexity : null,
    dataset: r.dataset,
    model: r.model,
    method: r.method,
  }))

  return {
    runs, models, summary, chartPoints, loading,
    modelFilter, datasetFilter, setModelFilter, setDatasetFilter, loadTrends,
  }
}
