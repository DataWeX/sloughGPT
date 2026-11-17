import { Component, type ComponentType, type ReactNode } from 'react'

import { logger } from '@/lib/dev-log'
import { isNotFoundDigest, parseRedirectDigest } from './next-compat/navigation'

/** The contract every route's error.tsx default export speaks (Next error.tsx). */
export type RouteErrorProps = {
  error: Error & { digest?: string }
  reset: () => void
}

interface Props {
  children: ReactNode
  errorComponent?: ComponentType<RouteErrorProps>
}

interface State {
  error: Error | null
}

/**
 * Per-route error boundary — the React half of the error.tsx contract.
 *
 * Placement (vite-entry): OUTSIDE the route's Suspense, so a rejected lazy
 * chunk — thrown while the loading fallback is still showing, i.e. past the
 * Suspense — reaches this boundary. The AppReadySentinel stays INSIDE the
 * Suspense: a route that errors before its chunk commits never declares
 * readiness for it.
 *
 * Digest pass-through is mandatory: the next/navigation shims throw
 * redirect()/notFound() as errors during render, and RouteDigestBoundary at
 * shell level must receive them to navigate / render the 404. Catching them
 * here would turn every redirect() into an error page.
 *
 * Reporting: CustomErrorHandler inside error.tsx owns user-facing reporting
 * (addGlobalError + reportError); componentDidCatch only logs the component
 * stack that error.tsx structurally cannot see.
 *
 * Reset: clears the error → children re-render → the route's Suspense shows
 * its loading skeleton again → the page re-attempts. (A rejected lazy chunk
 * caches its failure — "Try again" then re-throws the same error and the
 * error UI's Reload button does a full reload; a fresh-lazy retry is a
 * possible follow-up, not this card.)
 */
export class RouteErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: unknown): Partial<State> | null {
    if (parseRedirectDigest(error)) return null
    if (isNotFoundDigest(error)) return null
    return { error: error as Error }
  }

  componentDidCatch(error: Error, errorInfo: { componentStack?: string }) {
    logger.error('RouteErrorBoundary caught', {
      exception: error.message,
      stack: errorInfo.componentStack || '',
    })
  }

  reset = (): void => {
    this.setState({ error: null })
  }

  render() {
    const { error } = this.state
    if (error) {
      const { errorComponent: ErrorComponent } = this.props
      if (ErrorComponent) return <ErrorComponent error={error} reset={this.reset} />
      // Defensive: app/error.tsx covers every directory chain, so every
      // discovered route resolves an error component — this is for
      // unaccompanied trees only.
      return (
        <div className="flex min-h-[50vh] flex-col items-center justify-center gap-3 p-8 text-center">
          <p className="text-sm font-medium">Something went wrong</p>
          <p className="text-xs text-muted-foreground break-words max-w-md">{error.message}</p>
          <button
            type="button"
            onClick={this.reset}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
