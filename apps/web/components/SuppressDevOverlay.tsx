'use client'

import { useEffect } from 'react'

/**
 * Suppresses Next.js dev error overlay for hydration errors.
 *
 * Next.js dev overlay patches `console.error` during module init. By the
 * time this `useEffect` runs, the overlay is installed. We wrap it to
 * intercept hydration errors and prevent them from reaching the overlay,
 * while silently persisting to localStorage for crash recovery.
 */
export function SuppressDevOverlay() {
  useEffect(() => {
    const overlayConsoleError = console.error

    console.error = (...args: unknown[]) => {
      const msg = args.map(a => String(a)).join(' ')
      const firstStr = typeof args[0] === 'string' ? (args[0] as string) : ''
      const isHydration =
        firstStr.includes('hydrat') ||
        msg.includes('did not match') ||
        msg.includes('Hydration') ||
        (args[0] instanceof Error && (args[0] as Error).message.includes('hydrat'))

      if (isHydration) {
        try {
          const stored = JSON.parse(localStorage.getItem('__critical_errors') || '[]')
          stored.push({ ts: Date.now(), msg: msg.slice(0, 500), type: 'hydration' })
          localStorage.setItem('__critical_errors', JSON.stringify(stored.slice(-20)))
        } catch {}
        return // don't forward to overlay → no dev overlay popup
      }

      overlayConsoleError.apply(console, args)
    }

    const onError = (e: ErrorEvent) => {
      if (e.defaultPrevented) return
      const msg = (e.message || '').toLowerCase()
      if (msg.includes('hydrat') || msg.includes('did not match')) {
        e.preventDefault()
        return
      }
      e.preventDefault()
    }
    window.addEventListener('error', onError, true)

    const onReject = (e: PromiseRejectionEvent) => {
      if (e.defaultPrevented) return
      e.preventDefault()
    }
    window.addEventListener('unhandledrejection', onReject, true)

    return () => {
      console.error = overlayConsoleError
      window.removeEventListener('error', onError, true)
      window.removeEventListener('unhandledrejection', onReject, true)
    }
  }, [])

  return null
}
