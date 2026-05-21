'use client'

import { useEffect } from 'react'

/**
 * Suppresses Next.js dev error overlay for known non-fatal errors.
 *
 * The overlay hooks into `window.onerror` / `unhandledrejection` and
 * `console.error` — this wraps those to prevent the overlay for recoverable
 * patterns (fetch failures, network errors, WebGPU unavailable, etc.)
 * while still logging everything to the console.
 */
export function SuppressDevOverlay() {
  useEffect(() => {
    // Patch console.error to skip Next.js overlay propagation
    const origError = console.error
    console.error = (...args: unknown[]) => {
      const msg = args.join(' ')
      const quiet = [
        'fetch failed', 'Load failed', 'WebGPU', 'network',
        'Not found', '404', 'ECONNREFUSED', 'aborted',
        'ErrorBoundary caught',
      ]
      if (quiet.some(k => msg.includes(k))) {
        origError.apply(console, args)
        return
      }
      origError.apply(console, args)
    }

    // Suppress window.onerror for known non-fatal errors
    const onError = (e: ErrorEvent) => {
      const msg = e.message || ''
      if (msg.includes('fetch') || msg.includes('network') || msg.includes('404') || msg.includes('WebGPU')) {
        e.preventDefault()
      }
    }
    window.addEventListener('error', onError)

    // Suppress unhandled promise rejections for known non-fatal errors
    const onReject = (e: PromiseRejectionEvent) => {
      const msg = (e.reason?.message) || ''
      if (msg.includes('fetch') || msg.includes('network') || msg.includes('404') || msg.includes('WebGPU')) {
        e.preventDefault()
      }
    }
    window.addEventListener('unhandledrejection', onReject)

    return () => {
      console.error = origError
      window.removeEventListener('error', onError)
      window.removeEventListener('unhandledrejection', onReject)
    }
  }, [])

  return null
}
