'use client'

import { PageErrorHandler } from '@/components/PageErrorHandler'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'

export default function RegistryError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Registry" subtitle="Error" />} />
      <PageErrorHandler error={error} reset={reset} />
    </div>
  )
}
