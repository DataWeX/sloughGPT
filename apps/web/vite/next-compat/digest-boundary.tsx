import { Component, useEffect, type ReactNode } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { isNotFoundDigest, parseRedirectDigest, setViteNavigator } from './navigation'

type State = {
  redirectUrl: string | null
  notFound: boolean
  lastPath: string
}

/**
 * Catches digests thrown by next/navigation `redirect()` / `notFound()` shims.
 * Clears when router pathname moves so `<Navigate>` can run, then the shell
 * re-renders the destination route.
 */
class DigestBoundary extends Component<{ children: ReactNode; pathname: string }, State> {
  state: State = { redirectUrl: null, notFound: false, lastPath: this.props.pathname }

  static getDerivedStateFromError(error: unknown): Partial<State> | null {
    const url = parseRedirectDigest(error)
    if (url) return { redirectUrl: url, notFound: false }
    if (isNotFoundDigest(error)) return { notFound: true, redirectUrl: null }
    return null
  }

  static getDerivedStateFromProps(
    props: { pathname: string },
    state: State,
  ): Partial<State> | null {
    if (props.pathname === state.lastPath) return null
    if (state.redirectUrl || state.notFound) {
      return { redirectUrl: null, notFound: false, lastPath: props.pathname }
    }
    return { lastPath: props.pathname }
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
