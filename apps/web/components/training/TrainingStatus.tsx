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
    <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-3 space-y-2" role="alert" aria-live="assertive">
      <p className="text-sm font-medium text-destructive">Training failed</p>
      <p className="text-xs text-muted-foreground">{error}</p>
      <div className="flex gap-2">
        {onRetry && <Button size="sm" variant="outline" onClick={onRetry}>Retry</Button>}
        {onDismiss && (
          <Button size="sm" variant="ghost" onClick={onDismiss}>Dismiss</Button>
        )}
      </div>
    </div>
  )
}
