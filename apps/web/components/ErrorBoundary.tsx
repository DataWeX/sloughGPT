'use client'

import { Component, ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { IconAlert } from '@/components/ui'
import { addGlobalError } from '@/lib/error-store'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
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
                {this.state.error?.message || 'An unexpected error occurred'}
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
    throw error
  }
}