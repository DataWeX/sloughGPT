'use client'

import { useState, useEffect, useCallback } from 'react'
import { settingsController } from '@/lib/settings-controller'

export interface CompareRunData {
  run_id: string
  model: string
  dataset: string
  method: string
  final_loss: number
  perplexity: number
  quality_score: number
  training_time_s: number
  converged: boolean
}

export interface CompareResult {
  run_a: CompareRunData
  run_b: CompareRunData
  differences: Record<string, { run_a: unknown; run_b: unknown }>
  a_wins: number
  b_wins: number
}

export interface CompareRunOption {
  run_id: string
  model: string
  quality_score: number
}

const LOWER_IS_BETTER = new Set(['final_loss', 'perplexity', 'training_time_s'])

/** Which side wins a metric: 'a' | 'b' | 'tie'. Lower wins for loss/perplexity/time. */
export function compareMetric(field: string, valA: number, valB: number): 'a' | 'b' | 'tie' {
  if (LOWER_IS_BETTER.has(field)) {
    if (valA < valB) return 'a'
    if (valB < valA) return 'b'
    return 'tie'
  }
  if (valA > valB) return 'a'
  if (valB > valA) return 'b'
  return 'tie'
}

export interface UseRunCompareReturn {
  runOptions: CompareRunOption[]
  fetchingOptions: boolean
  runA: string
  runB: string
  setRunA: (v: string) => void
  setRunB: (v: string) => void
  result: CompareResult | null
  comparing: boolean
  compare: () => Promise<void>
  comparePair: (runA: string, runB: string) => Promise<CompareResult | null>
}

export function useRunCompare(): UseRunCompareReturn {
  const [runOptions, setRunOptions] = useState<CompareRunOption[]>([])
  const [fetchingOptions, setFetchingOptions] = useState(true)
  const [runA, setRunA] = useState('')
  const [runB, setRunB] = useState('')
  const [result, setResult] = useState<CompareResult | null>(null)
  const [comparing, setComparing] = useState(false)

  useEffect(() => {
    let active = true
    setFetchingOptions(true)
    settingsController
      .exportTrainingHistory('json', 50)
      .then(resp => {
        if (!active) return
        setRunOptions(
          (resp.outcomes ?? []).map((o: Record<string, unknown>) => ({
            run_id: o.run_id as string,
            model: o.model as string,
            quality_score: o.quality_score as number,
          })),
        )
      })
      .catch(err => console.error('Failed to fetch runs:', err))
      .finally(() => { if (active) setFetchingOptions(false) })
    return () => { active = false }
  }, [])

  const comparePair = useCallback(async (a: string, b: string) => {
    if (!a || !b) return null
    setComparing(true)
    try {
      const resp = await settingsController.compareTrainingRuns(a, b)
      const typed = resp as unknown as CompareResult
      setResult(typed)
      return typed
    } catch (err) {
      console.error('Failed to compare:', err)
      return null
    } finally {
      setComparing(false)
    }
  }, [])

  const compare = useCallback(() => comparePair(runA, runB).then(() => {}), [comparePair, runA, runB])

  return {
    runOptions, fetchingOptions, runA, runB, setRunA, setRunB,
    result, comparing, compare, comparePair,
  }
}
