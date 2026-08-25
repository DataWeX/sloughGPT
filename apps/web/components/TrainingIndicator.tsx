'use client'

import Link from 'next/link'
import { memo } from 'react'
import { cn } from '@sloughgpt/strui'
import { useTrainingSession } from '@/hooks/useTrainingSession'

export const TrainingIndicator = memo(function TrainingIndicator() {
  const { trainingRunning, phase, progress, loss, method } = useTrainingSession()

  if (!trainingRunning && phase !== 'complete' && phase !== 'error') {
    return null
  }

  return (
    <Link
      href="/training"
      prefetch={false}
      className={cn(
        'flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors',
        'hover:bg-primary/10',
        phase === 'error'
          ? 'text-destructive'
          : phase === 'complete'
            ? 'text-success'
            : 'text-primary',
      )}
      title="View training status"
    >
      <div className="relative flex h-2 w-2 shrink-0">
        {trainingRunning && (
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75" />
        )}
        <span
          className={cn(
            'relative inline-flex h-2 w-2 rounded-full',
            phase === 'error'
              ? 'bg-destructive'
              : phase === 'complete'
                ? 'bg-success'
                : 'bg-primary',
          )}
        />
      </div>
      <div className="min-w-0 flex-1" role="status" aria-live="polite">
        <div className="truncate text-xs font-medium">
          {phase === 'error'
            ? 'Training failed'
            : phase === 'complete'
              ? 'Training complete'
              : `${method === 'turbo' ? 'Turbo' : 'Training'}`}
        </div>
        {trainingRunning && (
          <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
            <span>{Math.round(progress ?? 0)}%</span>
            {loss != null && (
              <>
                <span>·</span>
                <span>loss {loss.toFixed(3)}</span>
              </>
            )}
          </div>
        )}
      </div>
    </Link>
  )
})
