'use client'

import { useMemo, useState } from 'react'
import type { Checkpoint } from '@/lib/souls-controller'

interface CheckpointFilterProps {
  checkpoints: Checkpoint[]
  onFiltered: (filtered: Checkpoint[]) => void
}

export function useCheckpointFilter(checkpoints: Checkpoint[]) {
  const [typeFilter, setTypeFilter] = useState<string>('all')
  const [lossMax, setLossMax] = useState<string>('')

  const filtered = useMemo(() => {
    let result = checkpoints

    if (typeFilter !== 'all') {
      result = result.filter(c => {
        const t = c.model_type || c.lineage || 'unknown'
        return t === typeFilter
      })
    }

    if (lossMax) {
      const max = parseFloat(lossMax)
      if (!isNaN(max)) {
        result = result.filter(c => c.loss != null && c.loss <= max)
      }
    }

    return result
  }, [checkpoints, typeFilter, lossMax])

  const types = useMemo(() => {
    const typeSet = new Set<string>()
    for (const c of checkpoints) {
      typeSet.add(c.model_type || c.lineage || 'unknown')
    }
    return [...typeSet].sort()
  }, [checkpoints])

  return { filtered, typeFilter, setTypeFilter, lossMax, setLossMax, types }
}
