'use client'

import { Button } from '@sloughgpt/strui'

/** Red error banner with retry/dismiss. */
export function TrainingErrorBanner({
  error,
  onRetry,
  onDismiss,
}: {
  error: string
  onRetry?: () => void
  onDismiss?: () => void
}) {
  return (
    <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-2.5 space-y-1.5" role="alert" aria-live="assertive">
      <p className="text-xs font-medium text-destructive">Training failed</p>
      <p className="text-[10px] text-muted-foreground/60">{error}</p>
      <div className="flex gap-1.5">
        {onRetry && <Button size="sm" variant="outline" className="h-6 text-[10px]" onClick={onRetry}>Retry</Button>}
        {onDismiss && (
          <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={onDismiss}>Dismiss</Button>
        )}
      </div>
    </div>
  )
}
