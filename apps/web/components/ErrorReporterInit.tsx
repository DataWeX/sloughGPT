'use client'

import { useEffect } from 'react'
import { initErrorReporter, reportError } from '@/lib/error-reporter'

export function ErrorReporterInit() {
  useEffect(() => {
    initErrorReporter()

    // Catch React hydration errors from the console
    const origError = console.error
    console.error = (...args: unknown[]) => {
      const joined = args.map(a => String(a)).join(' ')
      if (
        joined.includes('hydrat') ||
        joined.includes('did not match') ||
        joined.includes('Text content does not match')
      ) {
        reportError(
          joined.slice(0, 500),
          'hydration',
          args.slice(-1).length > 0
            ? { metadata: { detail: joined.slice(0, 1000) } }
            : undefined,
        )
        // Don't forward to native console.error — Next.js overlay hooks into it
        return
      }
      return origError.call(console, ...args)
    }

    return () => {
      console.error = origError
    }
  }, [])

  return null
}
