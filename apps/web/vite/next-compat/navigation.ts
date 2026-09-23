import {
  Navigate,
  useLocation,
  useNavigate,
  useParams as useReactRouterParams,
  useSearchParams as useReactRouterSearchParams,
} from 'react-router-dom'
import { useCallback, useMemo } from 'react'

export function usePathname(): string {
  return useLocation().pathname
}

export function useParams<
  T extends Record<string, string | string[] | undefined> = Record<string, string | undefined>,
>(): T {
  return useReactRouterParams() as T
}

type NextRouter = {
  push: (href: string, opts?: { scroll?: boolean }) => void
  replace: (href: string, opts?: { scroll?: boolean }) => void
  back: () => void
  forward: () => void
  refresh: () => void
  prefetch: (href: string) => Promise<void>
  pathname: string
  query: Record<string, string | string[] | undefined>
  asPath: string
  isReady: boolean
}

/** next/navigation useRouter → react-router navigate (app-router subset). */
export function useRouter(): NextRouter {
  const navigate = useNavigate()
  const location = useLocation()
  const push = useCallback(
    (href: string) => {
      navigate(href)
    },
    [navigate],
  )
  const replace = useCallback(
    (href: string) => {
      navigate(href, { replace: true })
    },
    [navigate],
  )
  const back = useCallback(() => navigate(-1), [navigate])
  const forward = useCallback(() => navigate(1), [navigate])
  const refresh = useCallback(() => window.location.reload(), [])
  const prefetch = useCallback(async () => {}, [])

  const search = location.search
  return useMemo(
    () => ({
      push,
      replace,
      back,
      forward,
      refresh,
      prefetch,
      pathname: location.pathname,
      query: Object.fromEntries(new URLSearchParams(search).entries()),
      asPath: location.pathname + search,
      isReady: true,
    }),
    [push, replace, back, forward, refresh, prefetch, location.pathname, search],
  )
}

/** next/navigation useSearchParams — same tuple API as react-router. */
export function useSearchParams(): ReturnType<typeof useReactRouterSearchParams> {
  return useReactRouterSearchParams()
}

export const REDIRECT_DIGEST_PREFIX = 'NEXT_REDIRECT;'
export const REDIRECT_MESSAGE_PREFIX = 'NEXT_REDIRECT:'

export const NOT_FOUND_DIGEST = 'NEXT_NOT_FOUND'
export const NOT_FOUND_MESSAGE = 'NEXT_NOT_FOUND'

type ViteNavigator = (url: string, opts?: { replace?: boolean }) => void
let viteNavigator: ViteNavigator | null = null

/** Registered by vite-entry so redirect() can navigate even if boundary recovery is delayed. */
export function setViteNavigator(fn: ViteNavigator | null): void {
  viteNavigator = fn
}

function scheduleViteRedirect(url: string): void {
  const nav = viteNavigator
  if (!nav) return
  queueMicrotask(() => {
    try {
      nav(url, { replace: true })
    } catch {
      /* navigate after unmount — ignore */
    }
  })
}

/** next/navigation redirect — throws a digest caught by RouteDigestBoundary. */
export function redirect(url: string, _status?: number | ResponseInit): never {
  scheduleViteRedirect(url)
  const err = new Error(`${REDIRECT_MESSAGE_PREFIX}${url}`) as Error & { digest?: string }
  err.digest = `${REDIRECT_DIGEST_PREFIX}${url}`
  throw err
}

/** next/navigation notFound — throws digest caught by RouteDigestBoundary. */
export function notFound(): never {
  const err = new Error(NOT_FOUND_MESSAGE) as Error & { digest?: string }
  err.digest = NOT_FOUND_DIGEST
  throw err
}

/**
 * Extract redirect URL from a thrown Next redirect (digest preferred, message fallback).
 */
export function parseRedirectDigest(error: unknown): string | null {
  if (!error || typeof error !== 'object') return null
  const e = error as { digest?: string; message?: string }
  if (e.digest && e.digest.startsWith(REDIRECT_DIGEST_PREFIX)) {
    return e.digest.slice(REDIRECT_DIGEST_PREFIX.length) || '/'
  }
  if (e.message && e.message.startsWith(REDIRECT_MESSAGE_PREFIX)) {
    return e.message.slice(REDIRECT_MESSAGE_PREFIX.length) || '/'
  }
  return null
}

export function isNotFoundDigest(error: unknown): boolean {
  if (!error || typeof error !== 'object') return false
  const e = error as { digest?: string; message?: string }
  return e.digest === NOT_FOUND_DIGEST || e.message === NOT_FOUND_MESSAGE
}

export { Navigate }
