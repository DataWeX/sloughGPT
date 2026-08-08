'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'

interface CheckpointFilterBarProps {
  types: string[]
  typeFilter: string
  onTypeFilterChange: (v: string) => void
  lossMax: string
  onLossMaxChange: (v: string) => void
  total: number
  shown: number
}

export function CheckpointFilterBar({ types, typeFilter, onTypeFilterChange, lossMax, onLossMaxChange, total, shown }: CheckpointFilterBarProps) {
  const hasFilters = typeFilter !== 'all' || lossMax !== ''

  if (total < 3) return null

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Filter checkpoints</CardTitle>
        <span className="text-[10px] text-muted-foreground/50">
          {shown}/{total} shown
        </span>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-3">
          <div className="space-y-1">
            <label className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">Type</label>
            <Select value={typeFilter} onValueChange={onTypeFilterChange}>
              <SelectTrigger className="h-7 text-[11px] w-32" aria-label="Filter by type">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All types</SelectItem>
                {types.map(t => (
                  <SelectItem key={t} value={t}>{t}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1">
            <label className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">Max loss</label>
            <Input
              type="number"
              value={lossMax}
              onChange={e => onLossMaxChange(e.target.value)}
              placeholder="e.g. 2.0"
              className="h-7 text-[11px] w-28"
              min="0"
              step="0.1"
            />
          </div>
          {hasFilters && (
            <div className="flex items-end">
              <button
                className="text-[10px] text-muted-foreground hover:text-foreground h-7"
                onClick={() => { onTypeFilterChange('all'); onLossMaxChange('') }}
              >
                Clear filters
              </button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
