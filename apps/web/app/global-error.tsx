'use client'

import { CustomErrorHandler } from '@/components/CustomErrorHandler'

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <html>
      <body className="bg-background text-foreground font-sans antialiased">
        <CustomErrorHandler error={error} reset={reset} />
      </body>
    </html>
  )
}
