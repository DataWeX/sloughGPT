'use client'

import { StatusBanner } from '@/components/composed/StatusBanner'

/** Re-export StatusBanner as TrainingErrorBanner for backward compatibility. */
export function TrainingErrorBanner({
  error,
  onRetry,
  onDismiss,
  onStop,
}: {
  error: string
  onRetry?: () => void
  onDismiss?: () => void
  onStop?: () => void
}) {
  return (
    <div className="flex items-center gap-1.5">
      <div className="flex-1">
        <StatusBanner
          variant="error"
          message={error}
          dismissible={false}
          onRetry={onRetry}
          onDismiss={onDismiss}
        />
      </div>
      {onStop && (
        <button
          type="button"
          className="underline text-destructive opacity-80 hover:opacity-100 text-[10px] shrink-0"
          onClick={onStop}
          aria-label="Stop"
        >
          Stop
        </button>
      )}
    </div>
  )
}
