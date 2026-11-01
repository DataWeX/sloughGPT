'use client'

import { PageErrorHandler } from '@/components/PageErrorHandler'
import { PageContainer } from '@/components/PageContainer'

export default function AuthError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <PageContainer title="Auth" subtitle="Error">
      <PageErrorHandler error={error} reset={reset} />
    </PageContainer>
  )
}
