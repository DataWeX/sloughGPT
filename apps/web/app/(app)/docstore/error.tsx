'use client'

import { PageErrorHandler } from '@/components/PageErrorHandler'

export default function DocstoreError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return <PageErrorHandler error={error} reset={reset} title="Docstore error" />
}
