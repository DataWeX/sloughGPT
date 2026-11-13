'use client'

import { Card, CardContent, Button, Badge, StatusBadge } from '@sloughgpt/strui'
import { Clock, BarChart3, Trash2, Tag, Star, Copy, CheckSquare, Square } from 'lucide-react'
import { formatDate, formatDuration, type TrainingRun } from '@/components/training/training-run'

function qualityBadge(score?: number) {
  if (score === undefined || score === null) return null
  const tone = score >= 0.8 ? 'success' : score >= 0.6 ? 'warning' : 'destructive'
  return <StatusBadge tone={tone}>{Math.round(score * 100)}%</StatusBadge>
}

interface RunListProps {
  loading: boolean
  runs: TrainingRun[]
  selectedRunId?: string
  selectedIds: Set<string>
  deleting: string | null
  onSelect: (run: TrainingRun) => void
  onToggleSelect: (runId: string) => void
  onToggleSelectAll: () => void
  onBookmark: (runId: string) => void
  onDuplicate: (runId: string) => void
  onDelete: (runId: string) => void
}

export function RunList({
  loading,
  runs,
  selectedRunId,
  selectedIds,
  deleting,
  onSelect,
  onToggleSelect,
  onToggleSelectAll,
  onBookmark,
  onDuplicate,
  onDelete,
}: RunListProps) {
  return (
    <div className="flex-1 space-y-2">
      {loading ? (
        Array.from({ length: 5 }).map((_, i) => (
          <Card key={i} className="animate-pulse">
            <CardContent className="py-4">
              <div className="h-4 bg-muted rounded w-1/3 mb-2" />
              <div className="h-3 bg-muted rounded w-2/3" />
            </CardContent>
          </Card>
        ))
      ) : runs.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            <BarChart3 className="h-12 w-12 mx-auto mb-3 opacity-30" />
            <p>No training runs found.</p>
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="flex items-center gap-2 mb-2">
            <button
              onClick={onToggleSelectAll}
              className="text-muted-foreground hover:text-foreground"
            >
              {selectedIds.size === runs.length && runs.length > 0 ? (
                <CheckSquare className="h-4 w-4" />
              ) : (
                <Square className="h-4 w-4" />
              )}
            </button>
            <span className="text-xs text-muted-foreground">Select all</span>
          </div>
          {runs.map((run) => (
            <Card
              key={run.run_id}
              className={`hover:shadow-sm transition-shadow cursor-pointer ${selectedRunId === run.run_id ? 'ring-2 ring-primary' : ''} ${selectedIds.has(run.run_id) ? 'bg-muted/30' : ''}`}
              onClick={() => {
                onSelect(run)
              }}
            >
              <CardContent className="py-3 flex items-center gap-4">
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onToggleSelect(run.run_id)
                  }}
                  className="flex-shrink-0 text-muted-foreground hover:text-foreground"
                >
                  {selectedIds.has(run.run_id) ? (
                    <CheckSquare className="h-4 w-4" />
                  ) : (
                    <Square className="h-4 w-4" />
                  )}
                </button>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-medium text-sm truncate">
                      {run.model || 'unknown'}
                    </span>
                    {run.method && (
                      <Badge variant="secondary" className="text-xs">
                        {run.method}
                      </Badge>
                    )}
                    {run.converged && <StatusBadge tone="success">converged</StatusBadge>}
                    {qualityBadge(run.quality_score)}
                    {(run.tags || []).map((tag) => (
                      <StatusBadge key={tag} tone="info">
                        <Tag className="h-2.5 w-2.5 mr-0.5" />
                        {tag}
                      </StatusBadge>
                    ))}
                  </div>
                  <div className="flex items-center gap-4 text-xs text-muted-foreground mt-1">
                    <span className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {formatDate(run.timestamp)}
                    </span>
                    {run.training_time_s && <span>{formatDuration(run.training_time_s)}</span>}
                    {run.epochs && <span>{run.epochs} epochs</span>}
                    {run.final_loss !== undefined && run.final_loss > 0 && (
                      <span>loss: {run.final_loss.toFixed(4)}</span>
                    )}
                    {run.perplexity !== undefined && run.perplexity > 0 && (
                      <span>ppl: {run.perplexity.toFixed(2)}</span>
                    )}
                    {run.notes && (
                      <span className="italic truncate max-w-[200px]">
                        &quot;{run.notes}&quot;
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    size="sm"
                    variant="ghost"
                    className={`h-7 px-2 ${run.bookmarked ? 'text-warning' : 'text-muted-foreground'}`}
                    onClick={(e) => {
                      e.stopPropagation()
                      onBookmark(run.run_id)
                    }}
                  >
                    <Star className={`h-4 w-4 ${run.bookmarked ? 'fill-current' : ''}`} />
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-7 px-2 text-muted-foreground"
                    onClick={(e) => {
                      e.stopPropagation()
                      onDuplicate(run.run_id)
                    }}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-destructive hover:text-destructive"
                    onClick={(e) => {
                      e.stopPropagation()
                      onDelete(run.run_id)
                    }}
                    disabled={deleting === run.run_id}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </>
      )}
    </div>
  )
}
