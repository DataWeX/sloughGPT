'use client'

import { Button } from '@sloughgpt/strui'
import { useRouter } from 'next/navigation'

/** Animated progress bar with status text during training. */
export function TrainingProgress({ status }: { status: string }) {
  return (
    <div className="space-y-2" role="status" aria-live="polite" aria-label="Training progress">
      <p className="text-sm text-muted-foreground">{status}</p>
      <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
        <div className="h-full bg-primary animate-pulse rounded-full w-[40%]" />
      </div>
    </div>
  )
}

/** Green success banner after training completes. */
export function TrainingCompleteBanner({
  message,
  explanation,
  onReset,
  onLoad,
}: {
  message?: string
  explanation?: string
  onReset?: () => void
  onLoad?: () => void
}) {
  const router = useRouter()
  return (
    <div className="rounded-lg border border-success/20 bg-success/5 p-3 space-y-2">
      <p className="text-sm font-medium text-success">{message || 'Training complete!'}</p>
      {explanation && <p className="text-xs text-muted-foreground">{explanation}</p>}
      <div className="flex flex-wrap gap-2">
        {onLoad && <Button size="sm" onClick={onLoad}>Load for chat</Button>}
        <Button size="sm" onClick={() => router.push('/chat')}>Try in chat</Button>
        {onReset && (
          <Button size="sm" variant="ghost" onClick={onReset}>Train another</Button>
        )}
      </div>
    </div>
  )
}

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
    <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-3 space-y-2">
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
