'use client'

import { PageErrorHandler } from '@/components/PageErrorHandler'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'

export default function LearnError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {

  return <PageErrorHandler error={error} reset={reset} title="Learner" />
}
