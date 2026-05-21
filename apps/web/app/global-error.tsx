'use client'

import { useEffect } from 'react'
import { IconAlert } from '@/components/ui'
import { addGlobalError } from '@/lib/error-store'

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    addGlobalError(error, 'global-error.tsx')
  }, [error])

  return (
    <html>
      <body className="bg-background text-foreground font-sans antialiased">
        <div className="min-h-screen flex items-center justify-center p-6">
          <div className="max-w-lg w-full space-y-6">
            <div className="text-center space-y-3">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-destructive/10 text-destructive mb-2">
                <IconAlert className="w-8 h-8" />
              </div>
              <h1 className="text-xl font-semibold">Critical error</h1>
              <p className="text-sm text-muted-foreground max-w-sm mx-auto">
                The application encountered a critical error. Please reload.
              </p>
            </div>
            <div className="flex items-center justify-center gap-3">
              <button
                onClick={() => reset()}
                className="inline-flex items-center justify-center rounded-md text-sm font-medium h-9 px-4 bg-primary text-primary-foreground hover:bg-primary/90"
              >
                Try again
              </button>
              <button
                onClick={() => window.location.href = '/'}
                className="inline-flex items-center justify-center rounded-md text-sm font-medium h-9 px-4 border border-input bg-background hover:bg-accent"
              >
                Go home
              </button>
            </div>
          </div>
        </div>
      </body>
    </html>
  )
}
