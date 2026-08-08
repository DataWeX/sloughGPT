'use client'

/**
 * ErrorLifecycle — consolidated error handler.
 *
 * Merges three previously separate components:
 *   - GlobalErrorHandler: window.error/unhandledrejection → error store + toast
 *   - ErrorReporterInit: window.error/unhandledrejection → POST /errors/log
 *   - SuppressDevOverlay: console.error → hydration localStorage + overlay suppression
 *
 * Single console.error wrapper, single set of window listeners, one non-fatal pattern list.
 * Hydration errors → localStorage + backend (no overlay).
 * Non-fatal errors → toast only.
 * Fatal errors → error store + toast + backend report.
 */

import { useEffect, useRef } from 'react'
import { useErrorStore } from '@/lib/error-store'
import { useToastStore } from '@/lib/toast-store'
import { reportError } from '@/lib/error-reporter'
import { logger } from '@/lib/dev-log'
import { chatDB } from '@/lib/db'

const _log = logger.child('error-lifecycle')

// ── Non-fatal patterns (merged from all three old components) ─────────

const NON_FATAL_PATTERNS = [
  /resizeobserver/i,
  /hydration/i,
  /suppresshydrationwarning/i,
  /react-hydration-error/i,
  /text content does not match server-rendered/i,
  /text content did not match/i,
  /aborterror/i,
  /cancelled/i,
  /network error/i,
  /fetch failed/i,
  /failed to fetch/i,
  /loadable/i,
  /chunk/i,
  /next-router/i,
  /useRouter/i,
]

function isNonFatal(message: string): boolean {
  return NON_FATAL_PATTERNS.some(p => p.test(message))
}

function isHydration(message: string, args?: unknown[]): boolean {
  if (message.includes('hydrat') || message.includes('did not match') || message.includes('Text content does not match')) return true
  if (args?.[0] instanceof Error && (args[0] as Error).message.includes('hydrat')) return true
  return false
}

// ── Stack extraction ──────────────────────────────────────────────────

function extractStackInfo(error: ErrorEvent | PromiseRejectionEvent): { file: string; line: number; col: number; stack: string } {
  let file = '', line = 0, col = 0, stack = ''

  if (error instanceof ErrorEvent) {
    file = error.filename || ''
    line = error.lineno || 0
    col = error.colno || 0
    if (error.error?.stack) stack = error.error.stack
  } else if (error instanceof PromiseRejectionEvent) {
    const reason = error.reason
    if (reason instanceof Error) {
      stack = reason.stack || ''
      const m = stack.match(/(?:https?:\/\/[^\s]+|\(file:\/\/[^\s]+|[a-zA-Z]:\\(?:[^\\]+\\)*[^:]+)/)
      file = m ? m[0].replace(/^\(/, '') : ''
      const lm = stack.match(/:(\d+):\d+/)
      if (lm) line = parseInt(lm[1], 10)
    } else if (typeof reason === 'string') {
      stack = reason
    } else if (reason && typeof reason === 'object') {
      try { stack = JSON.stringify(reason).slice(0, 500) } catch { stack = String(reason) }
    }
  }

  return { file, line, col, stack }
}

// ── Component ─────────────────────────────────────────────────────────

export function ErrorLifecycle() {
  const addError = useErrorStore(s => s.addError)
  const addToast = useToastStore(s => s.addToast)
  const initialized = useRef(false)

  useEffect(() => {
    if (initialized.current) return
    initialized.current = true

    // ── 1. Single console.error wrapper ──────────────────────────────
    const origConsoleError = console.error
    console.error = (...args: unknown[]) => {
      const msg = args.map(a => String(a)).join(' ')

      // Hydration errors: persist to Dexie + report to backend, suppress overlay
      if (isHydration(msg, args)) {
        chatDB.addError(msg.slice(0, 500), 'hydration').catch(() => /* DB write failed — non-critical */ {})
        reportError(msg.slice(0, 500), 'hydration', { metadata: { detail: msg.slice(0, 1000) } })
        return // don't forward to Next.js overlay
      }

      // All other console.error calls: pass through
      return origConsoleError.call(console, ...args)
    }

    // ── 2. Single window.error listener ──────────────────────────────
    const handleError = (event: ErrorEvent) => {
      if (event.defaultPrevented) return
      const message = event.message || 'Unknown error'

      if (isNonFatal(message)) {
        addToast(message, 'info')
        event.preventDefault()
        return
      }

      const { file, line, col, stack } = extractStackInfo(event)
      const source = file ? `${file}:${line}:${col}` : 'client'

      addError(event.error || event, { source, title: 'Runtime Error' })

      const verboseParts: string[] = []
      if (file) verboseParts.push(`at ${file}:${line}:${col}`)
      if (stack) verboseParts.push(stack.split('\n').slice(0, 4).map(s => s.trim()).join('\n'))
      const verbose = verboseParts.length > 0 ? verboseParts.join('\n') : undefined

      addToast('Something went wrong.', 'error', verbose)
      reportError(message, 'window.onerror', { stack, url: source, line, col })
      event.preventDefault()
    }

    // ── 3. Single unhandledrejection listener ────────────────────────
    const handleRejection = (event: PromiseRejectionEvent) => {
      if (event.defaultPrevented) return
      const reason = event.reason
      const message = reason instanceof Error ? reason.message : typeof reason === 'string' ? reason : 'Unhandled Promise Rejection'

      if (isNonFatal(message)) {
        addToast(message, 'info')
        event.preventDefault()
        return
      }

      const { file, line, col, stack } = extractStackInfo(event)
      const source = file ? `${file}:${line}:${col}` : 'client'

      addError(reason || event, { source, title: 'Unhandled Rejection' })

      const verboseParts: string[] = []
      if (file) verboseParts.push(`at ${file}:${line}:${col}`)
      if (stack) verboseParts.push(stack.split('\n').slice(0, 4).map(s => s.trim()).join('\n'))
      const verbose = verboseParts.length > 0 ? verboseParts.join('\n') : undefined

      addToast('Something went wrong.', 'error', verbose)
      reportError(message, 'unhandledrejection', { stack, url: source })
      event.preventDefault()
    }

    // ── 4. Capture-phase hydration suppression (prevents Next.js overlay) ──
    const handleCaptureError = (e: ErrorEvent) => {
      if (e.defaultPrevented) return
      const msg = (e.message || '').toLowerCase()
      if (msg.includes('hydrat') || msg.includes('did not match')) {
        e.preventDefault()
      }
    }

    // Attach: capture phase first (for overlay suppression), then bubble phase (for error store)
    window.addEventListener('error', handleCaptureError, true)
    window.addEventListener('error', handleError)
    window.addEventListener('unhandledrejection', handleRejection)

    // ── 5. Init error-reporter's window listeners (for batched POST /errors/log) ──
    // error-reporter.ts installs its own window.error + unhandledrejection listeners
    // that batch and POST to /errors/log. We call initErrorReporter() here so it runs
    // exactly once, and its listeners coexist with ours (both fire, no conflict).
    import('@/lib/error-reporter').then(m => m.initErrorReporter())

    _log.info('Error lifecycle initialized')

    return () => {
      console.error = origConsoleError
      window.removeEventListener('error', handleCaptureError, true)
      window.removeEventListener('error', handleError)
      window.removeEventListener('unhandledrejection', handleRejection)
    }
  }, [addError, addToast])

  return null
}
