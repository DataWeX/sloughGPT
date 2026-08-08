'use client'

import { useMemo, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { SearchInput } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

interface TrainingSearchProps {
  checkpoints: Checkpoint[]
  onFiltered: (filtered: Checkpoint[]) => void
}

export function useTrainingSearch(checkpoints: Checkpoint[]) {
  const [query, setQuery] = useState('')

  const filtered = useMemo(() => {
    if (!query.trim()) return checkpoints

    const q = query.toLowerCase().trim()
    return checkpoints.filter(c => {
      if (c.name.toLowerCase().includes(q)) return true
      if (c.soul?.toLowerCase().includes(q)) return true
      if (c.model_type?.toLowerCase().includes(q)) return true
      if (c.lineage?.toLowerCase().includes(q)) return true
      if (c.training_dataset?.toLowerCase().includes(q)) return true
      if (c.tagline?.toLowerCase().includes(q)) return true
      if (c.description?.toLowerCase().includes(q)) return true
      if (c.tags?.some(t => t.toLowerCase().includes(q))) return true
      if (c.loss != null && q.match(/^-?\d+\.?\d*$/) && Math.abs(c.loss - parseFloat(q)) < 0.01) return true
      return false
    })
  }, [checkpoints, query])

  return { filtered, query, setQuery }
}

export function TrainingSearchBar({ query, onQueryChange, total, shown }: { query: string; onQueryChange: (v: string) => void; total: number; shown: number }) {
  if (total < 3) return null

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Search</CardTitle>
        {query && (
          <span className="text-[10px] text-muted-foreground/50">
            {shown}/{total} match
          </span>
        )}
      </CardHeader>
      <CardContent>
        <SearchInput
          value={query}
          onChange={onQueryChange}
          placeholder="Search checkpoints by name, type, dataset..."
          className="h-8 text-[11px]"
        />
      </CardContent>
    </Card>
  )
}
