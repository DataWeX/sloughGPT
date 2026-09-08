'use client'

import { StatusBanner } from '@/components/composed/StatusBanner'

/** Re-export StatusBanner as TrainingErrorBanner for backward compatibility. */
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
    <StatusBanner
      variant="error"
      message={error}
      dismissible={false}
      onRetry={onRetry}
      onDismiss={onDismiss}
    />
  )
}
