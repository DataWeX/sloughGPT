'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input, cn } from '@sloughgpt/strui'
import { Search, Trash2, ArrowUpDown } from 'lucide-react'

interface Dataset {
  id: string
  name: string
  size: number
  row_count: number
  updated_at: string
  tags?: string[]
}

type SortField = 'date' | 'size' | 'name'

interface DatasetListCardProps {
  datasets: Dataset[]
  loading: boolean
  search: string
  sortBy: SortField
  onSearchChange: (value: string) => void
  onSortChange: (field: SortField) => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
}

const sortButtons: { field: SortField; label: string }[] = [
  { field: 'date', label: 'Date' },
  { field: 'size', label: 'Size' },
  { field: 'name', label: 'Name' },
]

export function DatasetListCard({
  datasets,
  loading,
  search,
  sortBy,
  onSearchChange,
  onSortChange,
  onSelect,
  onDelete,
}: DatasetListCardProps) {
  return (
    <Card data-testid="dataset-list">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Datasets</CardTitle>
          <div className="flex items-center gap-1">
            {sortButtons.map(s => (
              <Button
                key={s.field}
                size="sm"
                variant={sortBy === s.field ? 'default' : 'ghost'}
                className="text-[9px]"
                onClick={() => onSortChange(s.field)}
              >
                {s.label}
              </Button>
            ))}
          </div>
        </div>
        <div className="relative">
          <Search className="absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search datasets..."
            value={search}
            onChange={e => onSearchChange(e.target.value)}
            className="pl-7 h-8 text-xs"
            data-testid="dataset-search"
          />
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="h-16 rounded-lg bg-muted animate-pulse" />
            ))}
          </div>
        ) : datasets.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-6">No datasets found.</p>
        ) : (
          <div className="space-y-1.5 max-h-96 overflow-y-auto">
            {datasets.map(d => (
              <div
                key={d.id}
                className={cn(
                  'group flex items-center gap-3 rounded-lg border p-3 transition-colors cursor-pointer',
                  'border-border/60 hover:bg-muted/50'
                )}
                onClick={() => onSelect(d.id)}
                data-testid="dataset-entry"
              >
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium truncate">{d.name}</p>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-[10px] text-muted-foreground">{formatSize(d.size)}</span>
                    <span className="text-[10px] text-muted-foreground">{d.row_count.toLocaleString()} rows</span>
                    <span className="text-[10px] text-muted-foreground">{d.updated_at}</span>
                  </div>
                  {d.tags && d.tags.length > 0 && (
                    <div className="flex gap-1 mt-1">
                      {d.tags.map(t => (
                        <span key={t} className="text-[9px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{t}</span>
                      ))}
                    </div>
                  )}
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  className="opacity-0 group-hover:opacity-100 h-7 text-destructive"
                  onClick={e => { e.stopPropagation(); onDelete(d.id) }}
                  aria-label={`Delete ${d.name}`}
                >
                  <Trash2 className="h-3 w-3" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
