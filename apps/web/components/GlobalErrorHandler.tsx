'use client'

import { useEffect, useRef } from 'react'
import { useErrorStore } from '@/lib/error-store'
import { useToastStore } from '@/lib/toast-store'

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
      const { file, line, col, stack } = extractStackInfo(event)
      const message = event.message || 'Unknown error'
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
      const { file, line, col, stack } = extractStackInfo(event)
      const reason = event.reason
      const message = reason instanceof Error ? reason.message : typeof reason === 'string' ? reason : 'Unhandled Promise Rejection'
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
