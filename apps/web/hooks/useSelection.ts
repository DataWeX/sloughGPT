'use client'

import { useCallback, useState } from 'react'

/** Toggle an item in a Set state. */
export function toggleSetItem<T>(prev: Set<T>, id: T): Set<T> {
  const next = new Set(prev)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  return next
}

/**
 * Selection state hook — manages a Set<string> of selected IDs.
 *
 * Usage:
 *   const { selectedIds, toggleSelect, clearSelection } = useSelection()
 */
export function useSelection(initial?: Set<string>) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(initial ?? new Set())

  const toggleSelect = useCallback((id: string) => {
    setSelectedIds(prev => toggleSetItem(prev, id))
  }, [])

  const clearSelection = useCallback(() => {
    setSelectedIds(new Set())
  }, [])

  return { selectedIds, setSelectedIds, toggleSelect, clearSelection }
}
