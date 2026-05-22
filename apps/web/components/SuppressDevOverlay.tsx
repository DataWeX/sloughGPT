'use client'

import { useEffect } from 'react'

/**
 * Suppresses Next.js dev error overlay — all errors go to our own
 * ErrorBoundary / CustomErrorHandler instead of blocking the screen.
 *
 * Next.js dev overlay hooks into `window.onerror`, `unhandledrejection`,
 * and `console.error`. This intercepts all three and prevents the overlay
 * while still logging to console and letting our error reporter catch them.
 */
export function SuppressDevOverlay() {
  useEffect(() => {
    const origError = console.error
    console.error = (...args: unknown[]) => origError.apply(console, args)

    const onError = (e: ErrorEvent) => {
      if (e.defaultPrevented) return
      e.preventDefault()
    }
    window.addEventListener('error', onError, true)

    const onReject = (e: PromiseRejectionEvent) => {
      if (e.defaultPrevented) return
      e.preventDefault()
    }
    window.addEventListener('unhandledrejection', onReject, true)

    return () => {
      console.error = origError
      window.removeEventListener('error', onError, true)
      window.removeEventListener('unhandledrejection', onReject, true)
    }
  }, [])

  return null
}
