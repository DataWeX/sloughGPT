'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, cn } from '@sloughgpt/strui'

interface Experiment {
  id: string
  name?: string
  created?: string
  runs?: number
  status?: string
}

interface ExperimentListCardProps {
  experiments: Experiment[]
  selectedId?: string | null
  search?: string
  onSelect?: (id: string) => void
  onDelete?: (id: string) => void
  onSearch?: (query: string) => void
}

const STATUS_COLORS: Record<string, string> = {
  running: 'bg-primary/15 text-primary',
  completed: 'bg-success/15 text-success',
  failed: 'bg-destructive/15 text-destructive',
  queued: 'bg-warning/15 text-warning',
}

export function ExperimentListCard({ experiments, selectedId, search, onSelect, onDelete, onSearch }: ExperimentListCardProps) {
  const [localSearch, setLocalSearch] = useState(search ?? '')

  const filtered = localSearch
    ? experiments.filter(e => (e.id + (e.name ?? '')).toLowerCase().includes(localSearch.toLowerCase()))
    : experiments

  return (
    <Card data-testid="experiment-list">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">
            Experiments
            <span className="text-muted-foreground font-normal ml-2">({experiments.length})</span>
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="mb-3">
          <Input
            value={localSearch}
            onChange={e => { setLocalSearch(e.target.value); onSearch?.(e.target.value) }}
            placeholder="Search experiments..."
            data-testid="experiment-search"
          />
        </div>
        {filtered.length === 0 ? (
          <div className="text-sm text-muted-foreground text-center py-4">
            {experiments.length === 0 ? 'No experiments yet.' : 'No matches.'}
          </div>
        ) : (
          <div className="space-y-1">
            {filtered.map(exp => (
              <div
                key={exp.id}
                className={cn(
                  'flex items-center justify-between p-2 rounded border transition-colors',
                  selectedId === exp.id ? 'border-primary/50 bg-primary/5' : 'border-border hover:border-primary/30'
                )}
                data-testid={`experiment-${exp.id}`}
              >
                <button
                  className="flex-1 text-left min-w-0"
                  onClick={() => onSelect?.(exp.id)}
                >
                  <div className="text-xs font-medium truncate">{exp.name ?? exp.id}</div>
                  <div className="flex items-center gap-2 mt-0.5">
                    {exp.status && (
                      <span className={cn('text-[9px] px-1.5 py-0.5 rounded font-medium', STATUS_COLORS[exp.status] ?? 'bg-muted text-muted-foreground')}>
                        {exp.status}
                      </span>
                    )}
                    {exp.runs != null && (
                      <span className="text-[9px] text-muted-foreground">{exp.runs} runs</span>
                    )}
                    {exp.created && (
                      <span className="text-[9px] text-muted-foreground">{new Date(exp.created).toLocaleDateString()}</span>
                    )}
                  </div>
                </button>
                {onDelete && (
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 text-[10px] text-destructive shrink-0"
                    onClick={() => onDelete(exp.id)}
                  >
                    Del
                  </Button>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
