'use client'

import { PageErrorHandler } from '@/components/PageErrorHandler'

export default function SessionError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return <PageErrorHandler error={error} reset={reset} title="Session error" />
}
