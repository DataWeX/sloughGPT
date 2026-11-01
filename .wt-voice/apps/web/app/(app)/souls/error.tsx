'use client'

import { PageErrorHandler } from '@/components/PageErrorHandler'
import { PageContainer } from '@/components/PageContainer'

export default function SoulsError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <PageContainer title="Souls" subtitle="Error">
      <PageErrorHandler error={error} reset={reset} />
    </PageContainer>
  )
}
