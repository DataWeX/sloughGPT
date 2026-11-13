'use client'

import { Component, type ReactNode } from 'react'
import { Button } from '../ui/button'
import { IconAlert } from '../ui/icons'
import { Card, CardContent } from '../ui/card'

// ── ErrorBoundary (full-page) ──────────────────────────────────────

export interface ErrorBoundaryProps {
  children: ReactNode
  /** Custom fallback UI to render on error */
  fallback?: ReactNode
  /** Callback when an error is caught */
  onError?: (error: Error, errorInfo: { componentStack?: string }) => void
}

interface ErrorBoundaryState {
  hasError: boolean
  error?: Error
}

/**
 * Full-page React error boundary.
 *
 * Catches JavaScript errors in child components and renders a
 * fallback UI. Provides reload/home navigation options.
 *
 * @example
 * ```tsx
 * <ErrorBoundary>
 *   <App />
 * </ErrorBoundary>
 * ```
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: { componentStack?: string }) {
    this.props.onError?.(error, errorInfo)
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
              <Button variant="outline" onClick={() => (window.location.href = '/')}>
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

// ── SectionErrorBoundary (card-level) ──────────────────────────────

export interface SectionErrorBoundaryProps {
  children: ReactNode
  /** Name of the section for the error message */
  sectionName?: string
  /** Callback when an error is caught */
  onError?: (error: Error) => void
}

interface SectionErrorBoundaryState {
  hasError: boolean
  error?: Error
}

/**
 * Card-level React error boundary.
 *
 * Catches JavaScript errors in child components and renders
 * an inline error card with retry functionality. Suitable for
 * wrapping individual dashboard sections or cards.
 *
 * @example
 * ```tsx
 * <SectionErrorBoundary sectionName="Metrics">
 *   <MetricsCard />
 * </SectionErrorBoundary>
 * ```
 */
export class SectionErrorBoundary extends Component<SectionErrorBoundaryProps, SectionErrorBoundaryState> {
  constructor(props: SectionErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): SectionErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, _errorInfo: { componentStack?: string }) {
    this.props.onError?.(error)
  }

  reset() {
    this.setState({ hasError: false, error: undefined })
  }

  render() {
    if (this.state.hasError) {
      return (
        <Card className="border-destructive/30">
          <CardContent className="p-4 text-center space-y-2">
            <p className="text-sm text-destructive font-medium">
              {this.props.sectionName ? `${this.props.sectionName} failed to load` : 'Section failed to load'}
            </p>
            <p className="text-xs text-muted-foreground truncate max-w-xs mx-auto" title={this.state.error?.message}>
              {this.state.error?.message || 'An unexpected error occurred'}
            </p>
            <Button size="sm" variant="outline" onClick={() => this.reset()}>
              Retry
            </Button>
          </CardContent>
        </Card>
      )
    }
    return this.props.children
  }
}
