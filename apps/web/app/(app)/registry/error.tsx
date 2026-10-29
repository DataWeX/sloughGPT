'use client'

import { PageErrorHandler } from '@/components/PageErrorHandler'
import { PageContainer } from '@/components/PageContainer'

export default function RegistryError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <PageContainer title="Registry" subtitle="Error">
      <PageErrorHandler error={error} reset={reset} />
    </PageContainer>
  )
}
