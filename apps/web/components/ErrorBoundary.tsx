'use client'

import { Component, ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { IconAlert } from '@/components/ui'
import { addGlobalError } from '@/lib/error-store'
import { useToastStore } from '@/lib/toast-store'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
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
  return NON_FATAL_ERROR_PATTERNS.some(pattern => pattern.test(error.message))
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
    // Don't trigger full-page error for non-fatal errors
    if (isNonFatalError(error)) {
      // Log to toast instead
      try {
        useToastStore.getState().addToast('Something went wrong. Try refreshing the page.', 'error')
      } catch {
        // Toast store might not be initialized
      }
      return { hasError: false, error }
    }
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: { componentStack?: string }) {
    console.error('ErrorBoundary caught:', error, errorInfo)
    try {
      addGlobalError(error, errorInfo.componentStack || 'componentDidCatch')
    } catch {
      // store might not be initialized
    }
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div className="min-h-screen flex items-center justify-center bg-background p-6">
          <div className="max-w-lg w-full space-y-6">
            <div className="text-center space-y-3">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-destructive/10 text-destructive mb-2">
                <IconAlert className="w-8 h-8" />
              </div>
              <h1 className="text-xl font-semibold">Something went wrong</h1>
              <p className="text-sm text-muted-foreground max-w-sm mx-auto">
                Something went wrong. Try refreshing the page.
              </p>
            </div>
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

export function useErrorHandler() {
  return (error: Error) => {
    console.error('Error:', error)
    // Show as toast for non-fatal errors
    if (isNonFatalError(error)) {
      useToastStore.getState().addToast('Something went wrong. Try refreshing the page.', 'error')
    } else {
      throw error
    }
  }
}