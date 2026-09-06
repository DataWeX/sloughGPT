'use client'

import { Component, ReactNode } from 'react'
import { Button } from '@sloughgpt/strui'
import { IconAlert } from '@sloughgpt/strui'
import { addGlobalError } from '@/lib/error-store'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage, formatStackTrace, getErrorType } from '@/lib/error-utils'
import { logger } from '@/lib/dev-log'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
  showDetails: boolean
}

// Errors that should NOT trigger full-page error handler
const NON_FATAL_ERROR_PATTERNS = [
  /resizeobserver/i,
  /hydration/i,
  /suppresshydrationwarning/i,
  /aborterror/i,
  /cancelled/i,
  /network error/i,
  /fetch failed/i,
]

function isNonFatalError(error: Error): boolean {
  return NON_FATAL_ERROR_PATTERNS.some(pattern => pattern.test(error.message || ''))
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, showDetails: false }
  }

  static getDerivedStateFromError(error: Error): State {
    // Don't trigger full-page error for non-fatal errors
    if (isNonFatalError(error)) {
      // Log to toast instead
      try {
        useToastStore.getState().addToast('Something went wrong. Try refreshing the page.', 'error')
      } catch {
        logger.warning('ErrorBoundary: toast store not initialized', { error: error.message })
      }
      return { hasError: false, error, showDetails: false }
    }
    return { hasError: true, error, showDetails: false }
  }

  componentDidCatch(error: Error, errorInfo: { componentStack?: string }) {
    logger.error('ErrorBoundary caught', { exception: error.message, stack: errorInfo.componentStack || '' })
    try {
      addGlobalError(error, errorInfo.componentStack || 'componentDidCatch')
    } catch {
      logger.error('ErrorBoundary: failed to persist error', { error: error.message })
    }
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      const error = this.state.error
      const errorMessage = error ? extractErrorMessage(error, 'No error message') : 'Something went wrong'
      const errorType = error ? getErrorType(error) : null
      const stackFrames = error ? formatStackTrace(error.stack) : []

      return (
        <div className="min-h-screen flex items-center justify-center bg-background p-6">
          <div className="max-w-lg w-full space-y-6">
            <div className="text-center space-y-3">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-destructive/10 text-destructive mb-2">
                <IconAlert className="w-8 h-8" />
              </div>
              <h1 className="text-xl font-semibold flex items-center justify-center gap-2">
                Something went wrong
                {errorType && (
                  <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-destructive/10 text-destructive">
                    {errorType}
                  </span>
                )}
              </h1>
              <p className="text-sm text-muted-foreground max-w-sm mx-auto break-words">
                {errorMessage}
              </p>
            </div>

            {error && (
              <div className="text-center">
                <button
                  type="button"
                  onClick={() => this.setState({ showDetails: !this.state.showDetails })}
                  className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {this.state.showDetails ? 'Hide details' : 'Show details'}
                </button>
              </div>
            )}

            {this.state.showDetails && error && (
              <div className="rounded-md bg-muted p-3 text-xs font-mono max-h-48 overflow-y-auto">
                <pre className="whitespace-pre-wrap break-all text-muted-foreground">
                  {stackFrames.length > 0
                    ? stackFrames.map((frame, i) => <div key={i}>{frame}</div>)
                    : error.stack || error.message
                  }
                </pre>
              </div>
            )}

            <div className="flex items-center justify-center gap-3">
              <Button onClick={() => window.location.reload()}>
                Reload page
              </Button>
              <Button variant="outline" onClick={() => window.location.href = '/'}>
                Go home
              </Button>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
