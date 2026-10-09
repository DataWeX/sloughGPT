'use client'

import { useLiveOperations, type Operation, type OpType } from './useLiveOperations'

export type { Operation, OpType }

/**
 * Primary hook for UI components that need to track and manage operations.
 *
 * Delegates to SSE-backed useLiveOperations for real-time updates.
 * Falls back to HTTP polling if the SSE stream fails.
 *
 * @example
 * ```tsx
 * function TrainingCard() {
 *   const { cancel, cancelAll, isActive } = useOperations('training')
 *
 *   return (
 *     <Button disabled={!isActive} onClick={() => cancelAll('training')}>
 *       Cancel Training
 *     </Button>
 *   )
 * }
 * ```
 */
export function useOperations(type?: OpType) {
  return useLiveOperations(type)
}
