'use client'

import { PageErrorHandler } from '@/components/PageErrorHandler'

export default function VmError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return <PageErrorHandler error={error} reset={reset} title="VM error" />
}
