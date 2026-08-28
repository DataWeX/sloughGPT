'use client'

import { cn, Button } from '@sloughgpt/strui'
import { IconRefresh, IconTrash } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import type { UseVisionStudioReturn } from './useVisionStudio'

interface HistoryTabProps {
  vs: UseVisionStudioReturn
}

export function HistoryTab({ vs }: HistoryTabProps) {
  return (
    <div className="space-y-4">
      {!vs.trainingReport && <Skeleton className="h-32" />}

      {vs.trainingReport && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-2">
            <div className="p-2 rounded bg-muted/30 border border-border/40 text-center">
              <div className="text-sm font-semibold">{vs.trainingReport.images_learned}</div>
              <div className="text-[10px] text-muted-foreground">Images learned</div>
            </div>
            <div className="p-2 rounded bg-muted/30 border border-border/40 text-center">
              <div className="text-sm font-semibold">{vs.trainingReport.vocab_size}</div>
              <div className="text-[10px] text-muted-foreground">Vocab size</div>
            </div>
            <div className="p-2 rounded bg-muted/30 border border-border/40 text-center">
              <div className={cn(
                'text-sm font-semibold',
                vs.trainingReport.mean_accuracy >= 80 ? 'text-success' : vs.trainingReport.mean_accuracy >= 50 ? 'text-warning' : 'text-muted-foreground',
              )}>
                {vs.trainingReport.mean_accuracy.toFixed(1)}%
              </div>
              <div className="text-[10px] text-muted-foreground">Mean accuracy</div>
            </div>
            <div className="p-2 rounded bg-muted/30 border border-border/40 text-center">
              <div className={cn(
                'text-sm font-semibold',
                vs.trainingReport.last_accuracy >= 80 ? 'text-success' : vs.trainingReport.last_accuracy >= 50 ? 'text-warning' : 'text-muted-foreground',
              )}>
                {vs.trainingReport.last_accuracy.toFixed(1)}%
              </div>
              <div className="text-[10px] text-muted-foreground">Last accuracy</div>
            </div>
          </div>

          {vs.trainingReport.accuracy_history.length > 0 && (
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">Accuracy history</div>
              <div className="flex items-end gap-0.5 h-16">
                {vs.trainingReport.accuracy_history.map((a, i) => (
                  <div
                    key={i}
                    title={`${a.toFixed(1)}%`}
                    className={cn(
                      'flex-1 rounded-t transition-all',
                      a >= 80 ? 'bg-success/60' : a >= 50 ? 'bg-warning/60' : 'bg-muted-foreground/30',
                    )}
                    style={{ height: `${Math.max(a * 0.8, 4)}%` }}
                  />
                ))}
              </div>
            </div>
          )}

          {vs.trainingReport.caption_history.length > 0 && (
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">Recent captions learned</div>
              <ul className="space-y-1 max-h-40 overflow-y-auto">
                {vs.trainingReport.caption_history.slice(-20).reverse().map((cap, i) => (
                  <li key={i} className="p-2 rounded bg-muted/20 border border-border/20 text-xs leading-relaxed">{cap}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="flex gap-2 pt-2">
            <Button
              size="sm"
              variant="outline"
              onClick={vs.refreshReport}
            >
              <IconRefresh className="h-3.5 w-3.5 mr-1" />
              Refresh
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="text-destructive border-destructive/30 hover:bg-destructive/10"
              onClick={vs.handleReset}
              disabled={vs.resetLoading}
            >
              <IconTrash className="h-3.5 w-3.5 mr-1" />
              {vs.resetLoading ? 'Resetting...' : 'Reset model'}
            </Button>
          </div>
        </>
      )}
    </div>
  )
}
