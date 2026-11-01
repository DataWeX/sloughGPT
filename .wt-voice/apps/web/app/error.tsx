'use client'

import { CustomErrorHandler } from '@/components/CustomErrorHandler'

export default function RootError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return <CustomErrorHandler error={error} reset={reset} />
}
