'use client'

import { PageErrorHandler } from '@/components/PageErrorHandler'
import { PageContainer } from '@/components/PageContainer'

export default function FilesError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <PageContainer title="Files" subtitle="Error">
      <PageErrorHandler error={error} reset={reset} />
    </PageContainer>
  )
}
