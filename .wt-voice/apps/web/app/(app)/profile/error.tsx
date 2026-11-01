'use client'

import { PageErrorHandler } from '@/components/PageErrorHandler'

export default function ProfileError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return <PageErrorHandler error={error} reset={reset} title="Profile error" />
}
