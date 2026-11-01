'use client'

import { PageErrorHandler } from '@/components/PageErrorHandler'
import { PageContainer } from '@/components/PageContainer'

export default function FeedbackError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <PageContainer title="Feedback" subtitle="Error">
      <PageErrorHandler error={error} reset={reset} />
    </PageContainer>
  )
}
