'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

interface TrainingQuickActionsProps {
  checkpoints: Checkpoint[]
  onLoadBest?: (name: string) => void
  onExportMetrics?: () => void
  onExportNotes?: () => void
  onClearNotes?: () => void
}

function getBestCheckpoint(checkpoints: Checkpoint[]): Checkpoint | null {
  const valid = checkpoints.filter(c => c.loss != null && c.loss > 0)
  if (!valid.length) return null
  return valid.reduce((a, b) => (a.loss ?? Infinity) < (b.loss ?? Infinity) ? a : b)
}

function getRecentCount(checkpoints: Checkpoint[]): number {
  const oneDayAgo = Date.now() - 86400000
  return checkpoints.filter(c => {
    if (!c.born_at) return false
    try {
      return new Date(c.born_at).getTime() > oneDayAgo
    } catch {
      return false
    }
  }).length
}

export function TrainingQuickActions({ checkpoints, onLoadBest, onExportMetrics, onExportNotes, onClearNotes }: TrainingQuickActionsProps) {
  const best = getBestCheckpoint(checkpoints)
  const recentCount = getRecentCount(checkpoints)
  const hasActions = best || onExportMetrics || onExportNotes || onClearNotes

  if (!hasActions) return null

  return (
    <Card data-testid="training-quick-actions">
      <CardHeader className="flex flex-row items-center justify-between py-3">
        <CardTitle className="text-base">Quick Actions</CardTitle>
        <div className="flex items-center gap-2">
          {recentCount > 0 && (
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0 h-4">
              {recentCount} new today
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-2">
          {best && onLoadBest && (
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-[11px]"
              onClick={() => onLoadBest(best.name)}
              disabled={best.is_loaded}
            >
              {best.is_loaded ? 'Best loaded' : `Load best (${best.loss?.toFixed(3)})`}
            </Button>
          )}
          {onExportMetrics && (
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-[11px]"
              onClick={onExportMetrics}
            >
              Export metrics
            </Button>
          )}
          {onExportNotes && (
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-[11px]"
              onClick={onExportNotes}
            >
              Export notes
            </Button>
          )}
          {onClearNotes && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-[11px] text-destructive"
              onClick={onClearNotes}
            >
              Clear notes
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
