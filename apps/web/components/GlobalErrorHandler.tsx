'use client'

import { useEffect, useRef } from 'react'
import { useErrorStore } from '@/lib/error-store'
import { useToastStore } from '@/lib/toast-store'

// Errors that should NOT trigger full-page error handler
const NON_FATAL_ERROR_PATTERNS = [
  /resizeobserver/i,
  /hydration/i,
  /suppresshydrationwarning/i,
  /react-hydration-error/i,
  /text content does not match server-rendered/i,
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

function isNonFatalError(message: string): boolean {
  return NON_FATAL_ERROR_PATTERNS.some(pattern => pattern.test(message))
}

function extractStackInfo(error: ErrorEvent | PromiseRejectionEvent): { file: string; line: number; col: number; stack: string } {
  let file = ''
  let line = 0
  let col = 0
  let stack = ''

  if (error instanceof ErrorEvent) {
    file = error.filename || ''
    line = error.lineno || 0
    col = error.colno || 0
    if (error.error?.stack) stack = error.error.stack
  } else if (error instanceof PromiseRejectionEvent) {
    const reason = error.reason
    if (reason instanceof Error) {
      file = extractFileFromStack(reason.stack) || ''
      line = extractLineFromStack(reason.stack) || 0
      stack = reason.stack || ''
    } else if (typeof reason === 'string') {
      stack = reason
    } else if (reason && typeof reason === 'object') {
      try { stack = JSON.stringify(reason).slice(0, 500) } catch { stack = String(reason) }
    }
  }

  return { file, line, col, stack }
}

function extractFileFromStack(stack?: string): string {
  if (!stack) return ''
  const match = stack.match(/(?:https?:\/\/[^\s]+|\(file:\/\/[^\s]+|[a-zA-Z]:\\(?:[^\\]+\\)*[^:]+)/)
  return match ? match[0].replace(/^\(/, '') : ''
}

function extractLineFromStack(stack?: string): number {
  if (!stack) return 0
  const match = stack.match(/:(\d+):\d+/)
  return match ? parseInt(match[1], 10) : 0
}

export function GlobalErrorHandler() {
  const addError = useErrorStore(s => s.addError)
  const addToast = useToastStore(s => s.addToast)
  const initialized = useRef(false)

  useEffect(() => {
    if (initialized.current) return
    initialized.current = true

    const handleError = (event: ErrorEvent) => {
      const message = event.message || 'Unknown error'
      
      // Non-fatal errors: show as toast only, don't add to error store
      if (isNonFatalError(message)) {
        addToast(message, 'info')
        event.preventDefault()
        return
      }

      const { file, line, col, stack } = extractStackInfo(event)
      const source = file ? `${file}:${line}:${col}` : 'client'

      addError(event.error || event, { source, title: 'Runtime Error' })

      const verboseParts: string[] = []
      if (file) verboseParts.push(`at ${file}:${line}:${col}`)
      if (stack) {
        const lines = stack.split('\n').slice(0, 4).map(s => s.trim()).join('\n')
        verboseParts.push(lines)
      }
      const verbose = verboseParts.length > 0 ? verboseParts.join('\n') : undefined

      addToast(message, 'error', verbose)
      event.preventDefault()
    }

    const handleRejection = (event: PromiseRejectionEvent) => {
      const reason = event.reason
      const message = reason instanceof Error ? reason.message : typeof reason === 'string' ? reason : 'Unhandled Promise Rejection'
      
      // Non-fatal errors: show as toast only
      if (isNonFatalError(message)) {
        addToast(message, 'info')
        event.preventDefault()
        return
      }

      const { file, line, col, stack } = extractStackInfo(event)
      const source = file ? `${file}:${line}:${col}` : 'client'

      addError(reason || event, { source, title: 'Unhandled Rejection' })

      const verboseParts: string[] = []
      if (file) verboseParts.push(`at ${file}:${line}:${col}`)
      if (stack) {
        const lines = stack.split('\n').slice(0, 4).map(s => s.trim()).join('\n')
        verboseParts.push(lines)
      }
      const verbose = verboseParts.length > 0 ? verboseParts.join('\n') : undefined

      addToast(message, 'error', verbose)
      event.preventDefault()
    }

    window.addEventListener('error', handleError)
    window.addEventListener('unhandledrejection', handleRejection)

    return () => {
      window.removeEventListener('error', handleError)
      window.removeEventListener('unhandledrejection', handleRejection)
    }
  }, [addError, addToast])

  return null
}
