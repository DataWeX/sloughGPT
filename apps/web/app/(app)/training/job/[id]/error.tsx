'use client'

import { PageErrorHandler } from '@/components/PageErrorHandler'

export default function TrainingJobError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return <PageErrorHandler error={error} reset={reset} title="Training job error" />
}
