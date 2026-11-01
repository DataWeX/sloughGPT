'use client'

import { PageErrorHandler } from '@/components/PageErrorHandler'
import { PageContainer } from '@/components/PageContainer'

export default function LearnError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {

  return <PageErrorHandler error={error} reset={reset} title="Learner" />
}
