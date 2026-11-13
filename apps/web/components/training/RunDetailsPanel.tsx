'use client'

import { Card, CardContent, Button, Badge, Input } from '@sloughgpt/strui'
import { Star, Download, X, Tag, GitCompare, Plus } from 'lucide-react'
import { formatDuration, type TrainingRun } from '@/components/training/training-run'

interface RunDetailsPanelProps {
  run: TrainingRun
  runs: TrainingRun[]
  onBookmark: (runId: string) => void
  onExportSingle: (runId: string) => void
  onClose: () => void
  newTag: string
  onNewTagChange: (v: string) => void
  onAddTag: () => void
  onRemoveTag: (tag: string) => void
  editingNotes: boolean
  onEditNotes: () => void
  notesValue: string
  onNotesChange: (v: string) => void
  onSaveNotes: () => void
  onCancelEdit: () => void
  compareRunId: string
  onCompareIdChange: (v: string) => void
  compareResult: Record<string, unknown> | null
  onCompare: () => void
}

export function RunDetailsPanel({
  run,
  runs,
  onBookmark,
  onExportSingle,
  onClose,
  newTag,
  onNewTagChange,
  onAddTag,
  onRemoveTag,
  editingNotes,
  onEditNotes,
  notesValue,
  onNotesChange,
  onSaveNotes,
  onCancelEdit,
  compareRunId,
  onCompareIdChange,
  compareResult,
  onCompare,
}: RunDetailsPanelProps) {
  return (
    <Card className="w-80 flex-shrink-0">
      <CardContent className="pt-4 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-sm">Run Details</h3>
          <div className="flex items-center gap-1">
            <Button
              size="sm"
              variant="ghost"
              className={`h-6 px-2 ${run.bookmarked ? 'text-warning' : 'text-muted-foreground'}`}
              onClick={() => onBookmark(run.run_id)}
            >
              <Star className={`h-3 w-3 ${run.bookmarked ? 'fill-current' : ''}`} />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              className="h-6 px-2"
              onClick={() => onExportSingle(run.run_id)}
            >
              <Download className="h-3 w-3 mr-1" /> Export
            </Button>
            <button onClick={onClose}>
              <X className="h-4 w-4 text-muted-foreground" />
            </button>
          </div>
        </div>
        <div className="space-y-2 text-sm">
          <div>
            <span className="text-muted-foreground">ID:</span>{' '}
            <span className="font-mono text-xs">{run.run_id}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Model:</span> {run.model}
          </div>
          <div>
            <span className="text-muted-foreground">Method:</span> {run.method}
          </div>
          <div>
            <span className="text-muted-foreground">Dataset:</span> {run.dataset}
          </div>
          <div>
            <span className="text-muted-foreground">Epochs:</span> {run.epochs}
          </div>
          <div>
            <span className="text-muted-foreground">Final Loss:</span>{' '}
            {run.final_loss?.toFixed(4)}
          </div>
          <div>
            <span className="text-muted-foreground">Best Loss:</span>{' '}
            {run.best_loss?.toFixed(4)}
          </div>
          <div>
            <span className="text-muted-foreground">Perplexity:</span>{' '}
            {run.perplexity?.toFixed(2)}
          </div>
          <div>
            <span className="text-muted-foreground">Quality:</span>{' '}
            {run.quality_score ? `${Math.round(run.quality_score * 100)}%` : '-'}
          </div>
          <div>
            <span className="text-muted-foreground">Duration:</span>{' '}
            {formatDuration(run.training_time_s)}
          </div>
          <div>
            <span className="text-muted-foreground">Converged:</span>{' '}
            {run.converged ? 'Yes' : 'No'}
          </div>
          <div>
            <span className="text-muted-foreground">Early Stopped:</span>{' '}
            {run.early_stopped ? 'Yes' : 'No'}
          </div>
        </div>

        <div className="border-t pt-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Tags</span>
          </div>
          <div className="flex flex-wrap gap-1 mb-2">
            {(run.tags || []).map((tag) => (
              <Badge
                key={tag}
                className="bg-primary/15 text-primary text-xs flex items-center gap-1"
              >
                <Tag className="h-2.5 w-2.5" />
                {tag}
                <button onClick={() => onRemoveTag(tag)} className="ml-0.5 hover:text-primary">
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            ))}
          </div>
          <div className="flex gap-1">
            <Input
              placeholder="Add tag..."
              value={newTag}
              onChange={(e) => onNewTagChange(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && onAddTag()}
              className="text-xs h-7"
            />
            <Button
              size="sm"
              variant="outline"
              className="h-7 px-2"
              onClick={onAddTag}
              disabled={!newTag.trim()}
            >
              <Plus className="h-3 w-3" />
            </Button>
          </div>
        </div>

        <div className="border-t pt-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Notes</span>
            {!editingNotes && (
              <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={onEditNotes}>
                Edit
              </Button>
            )}
          </div>
          {editingNotes ? (
            <div className="space-y-2">
              <textarea
                value={notesValue}
                onChange={(e) => onNotesChange(e.target.value)}
                className="w-full text-xs border rounded p-2 min-h-[80px]"
                placeholder="Add notes about this training run..."
              />
              <div className="flex gap-1">
                <Button size="sm" className="h-6 text-xs" onClick={onSaveNotes}>
                  Save
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-6 text-xs"
                  onClick={onCancelEdit}
                >
                  Cancel
                </Button>
              </div>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">{run.notes || 'No notes yet.'}</p>
          )}
        </div>

        <div className="border-t pt-3">
          <div className="flex items-center gap-2 mb-2">
            <GitCompare className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="text-sm font-medium">Compare</span>
          </div>
          <div className="flex gap-1">
            <select
              value={compareRunId}
              onChange={(e) => onCompareIdChange(e.target.value)}
              className="text-xs border rounded px-2 py-1 flex-1"
            >
              <option value="">Select run to compare...</option>
              {runs
                .filter((r) => r.run_id !== run.run_id)
                .map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.model} - {r.run_id.slice(0, 12)}
                  </option>
                ))}
            </select>
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-xs"
              onClick={onCompare}
              disabled={!compareRunId}
            >
              Compare
            </Button>
          </div>
          {compareResult && (
            <div className="mt-2 text-xs space-y-1 bg-muted/50 rounded p-2">
              <div className="font-medium">Differences:</div>
              {Object.entries(
                (compareResult.differences as Record<
                  string,
                  { run_a: unknown; run_b: unknown }
                >) || {},
              ).map(([field, diff]) => (
                <div key={field} className="flex justify-between">
                  <span className="text-muted-foreground">{field}:</span>
                  <span>
                    {String(diff.run_a)} → {String(diff.run_b)}
                  </span>
                </div>
              ))}
              {compareResult.a_wins !== undefined && (
                <div className="pt-1 text-muted-foreground">
                  Score: {String(compareResult.a_wins)} vs {String(compareResult.b_wins)}
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
