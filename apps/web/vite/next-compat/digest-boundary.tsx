import { Component, useEffect, type ReactNode } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { logger } from '@/lib/dev-log'
import { isNotFoundDigest, parseRedirectDigest, setViteNavigator } from './navigation'

type State = {
  redirectUrl: string | null
  notFound: boolean
  lastPath: string
  error: Error | null
}

/**
 * Catches digests thrown by next/navigation `redirect()` / `notFound()` shims.
 * Clears when router pathname moves so `<Navigate>` can run, then the shell
 * re-renders the destination route.
 *
 * Also the shell-level LAST-resort boundary (vite-entry puts it above
 * AppLayout, where CommandPalette lives — outside every per-route
 * RouteErrorBoundary): non-digest crashes are captured into state and
 * rendered as a fallback. Returning null for them made render() re-emit
 * children, which re-threw on every retry — the "React will try to recreate
 * this component tree" loop of card 20261123_073036 (2,322 recreations).
 */
class DigestBoundary extends Component<{ children: ReactNode; pathname: string }, State> {
  state: State = {
    redirectUrl: null,
    notFound: false,
    lastPath: this.props.pathname,
    error: null,
  }

  static getDerivedStateFromError(error: unknown): Partial<State> | null {
    const url = parseRedirectDigest(error)
    if (url) return { redirectUrl: url, notFound: false, error: null }
    if (isNotFoundDigest(error)) return { notFound: true, redirectUrl: null, error: null }
    // Non-digest crash: capture it so render() shows the fallback instead
    // of children (which would throw again — the loop).
    return { error: error instanceof Error ? error : new Error(String(error)) }
  }

  static getDerivedStateFromProps(
    props: { pathname: string },
    state: State,
  ): Partial<State> | null {
    if (props.pathname === state.lastPath) return null
    if (state.redirectUrl || state.notFound || state.error) {
      // Path change clears a captured crash too: navigation is a fresh
      // attempt (shell-level crashes re-throw once and land back here).
      return { redirectUrl: null, notFound: false, error: null, lastPath: props.pathname }
    }
    return { lastPath: props.pathname }
  }

  componentDidCatch(error: Error, errorInfo: { componentStack?: string }) {
    // Digests are expected control flow (redirect()/notFound() shims) —
    // only real crashes belong in the log.
    if (parseRedirectDigest(error) || isNotFoundDigest(error)) return
    logger.error('RouteDigestBoundary caught', {
      exception: error.message,
      stack: errorInfo.componentStack || '',
    })
  }

  render() {
    if (this.state.redirectUrl) {
      return <Navigate to={this.state.redirectUrl} replace />
    }
    if (this.state.notFound) {
      return (
        <div className="flex min-h-[50vh] flex-col items-center justify-center gap-3 p-8 text-center">
          <p className="text-2xl font-semibold">404</p>
          <p className="text-sm text-muted-foreground">This page could not be found.</p>
        </div>
      )
    }
    const { error } = this.state
    if (error) {
      // Mirror RouteErrorBoundary's defensive fallback, with a full reload:
      // a shell-level tree has no "Try again" — its children are broken.
      return (
        <div className="flex min-h-[50vh] flex-col items-center justify-center gap-3 p-8 text-center">
          <p className="text-sm font-medium">Something went wrong</p>
          <p className="text-xs text-muted-foreground break-words max-w-md">{error.message}</p>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Reload
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

/** Location-aware wrapper (class boundary cannot call hooks). */
export function RouteDigestBoundary({ children }: { children: ReactNode }) {
  const { pathname } = useLocation()
  return <DigestBoundary pathname={pathname}>{children}</DigestBoundary>
}

/**
 * Registers `useNavigate` with the navigation shim so `redirect()` can fire
 * from a microtask while React recovers from the thrown error.
 */
export function ViteNavigatorBinder(): null {
  const navigate = useNavigate()
  useEffect(() => {
    setViteNavigator((url, opts) => {
      navigate(url, { replace: opts?.replace ?? true })
    })
    return () => setViteNavigator(null)
  }, [navigate])
  return null
}
