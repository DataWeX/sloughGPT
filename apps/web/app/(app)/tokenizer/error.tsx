'use client'

import { PageErrorHandler } from '@/components/PageErrorHandler'
import { PageContainer } from '@/components/PageContainer'

export default function TokenizerError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {

  return <PageErrorHandler error={error} reset={reset} title="Tokenizer" />
}
