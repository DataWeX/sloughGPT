'use client'

import { useState, useCallback } from 'react'
import { consciousnessController } from '@/lib/consciousness-controller'

export interface ConsciousnessBatchOperation {
  type: 'seed' | 'feedback' | 'reflect' | 'config'
  payload: Record<string, unknown>
}

/** Union of the per-operation payload types batch can settle with. */
type BatchOperationResult =
  | Awaited<ReturnType<typeof consciousnessController.seedData>>
  | Awaited<ReturnType<typeof consciousnessController.submitFeedback>>
  | Awaited<ReturnType<typeof consciousnessController.reflect>>
  | Awaited<ReturnType<typeof consciousnessController.updateConfig>>
  | null

export function useConsciousnessBatch() {
  const [results, setResults] = useState<PromiseSettledResult<BatchOperationResult>[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const executeBatch = useCallback(async (operations: ConsciousnessBatchOperation[]) => {
    setLoading(true)
    setError(null)
    try {
      const batchResults = await Promise.allSettled(
        operations.map(op => {
          switch (op.type) {
            case 'seed': return consciousnessController.seedData(op.payload)
            case 'feedback': return consciousnessController.submitFeedback(op.payload)
            case 'reflect': return consciousnessController.reflect()
            case 'config': return consciousnessController.updateConfig(op.payload)
            default: return Promise.resolve(null)
          }
        })
      )
      setResults(batchResults)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Batch request failed')
    } finally {
      setLoading(false)
    }
  }, [])

  return { results, loading, error, executeBatch }
}
