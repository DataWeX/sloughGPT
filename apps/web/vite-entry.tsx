/**
 * Phase 1 Vite entry — AppLayout shell + every app/(app) route via import.meta.glob.
 * Coexists with Next (index.html is ignored by next dev/build).
 * Phase 4: legacy REDIRECTS mount before discovered routes (proxy.ts shadowing).
 */
import './app/globals.css'

import { Suspense, StrictMode, lazy } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Link, Navigate, Outlet, Route, Routes } from 'react-router-dom'

import AppLayout from '@/components/AppLayout'
import { ThemeProvider } from '@/components/ThemeProvider'
import { ModelProvider } from '@/contexts/ModelContext'
import { LocaleProvider } from '@/hooks/useLocale'
import { ErrorLifecycle } from '@/components/ErrorLifecycle'
import { StartupOverlay } from '@/components/startup/StartupOverlay'
import { RouteDigestBoundary, ViteNavigatorBinder } from '@/vite/next-compat/digest-boundary'
import { AppReadySentinel } from '@/vite/AppReadySentinel'
import { discoveredRoutes } from '@/vite/routes'
import { REDIRECTS, splitTarget } from '@/lib/redirects'

// Dev-only design lab (Three.js dock concept showcase). Never cited as a
// journey — AGENTS.md Page Rule keeps product pages honest, and a design
// exploration is not one: mounted only in the dev shell (import.meta.env.DEV),
// lazy so three.js never loads in a production bundle, and absent from the
// Next app's page tree so the page-baseline ratchet is untouched.
const DesignLab3D = lazy(() => import('@/components/designlab/DesignLab3D'))

function PageFallback() {
  return (
    <div className="flex min-h-[40vh] items-center justify-center text-sm text-muted-foreground">
      Loading…
    </div>
  )
}

function Shell() {
  return (
    <RouteDigestBoundary>
      <ViteNavigatorBinder />
      <AppLayout>
        <Suspense fallback={<PageFallback />}>
          <Outlet />
        </Suspense>
      </AppLayout>
    </RouteDigestBoundary>
  )
}

function NotFound() {
  // Mirrors app/not-found.tsx (Next 404): h1 + Home/Chat links — the
  // not-found-page.cy.ts spec asserts this markup in both stacks.
  return (
    <div className="flex min-h-[50vh] flex-col items-center justify-center gap-4 p-8 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
        <span className="text-2xl font-bold text-primary">?</span>
      </div>
      <h1 className="text-xl font-semibold">Page not found</h1>
      <p className="text-sm text-muted-foreground text-center max-w-sm">
        The page you&apos;re looking for doesn&apos;t exist or has been moved.
      </p>
      <div className="flex gap-2 mt-2">
        <Link
          to="/"
          className="inline-flex items-center justify-center rounded-md border border-input bg-background px-3 h-8 text-xs font-medium hover:bg-accent hover:text-accent-foreground transition-colors"
        >
          Home
        </Link>
        <Link
          to="/chat"
          className="inline-flex items-center justify-center rounded-md bg-primary text-primary-foreground px-3 h-8 text-xs font-medium hover:bg-primary/90 transition-colors"
        >
          Chat
        </Link>
      </div>
    </div>
  )
}

function App() {
  const hasHome = discoveredRoutes.some((r) => r.path === '/')
  return (
    <ThemeProvider>
      <ModelProvider>
        <LocaleProvider>
          <ErrorLifecycle />
          <StartupOverlay />
          <BrowserRouter>
            <Routes>
              <Route element={<Shell />}>
                {/* Legacy redirects first — Next proxy shadows real pages (30/39 have page.tsx) */}
                {Object.entries(REDIRECTS).map(([from, to]) => {
                  const { path, search } = splitTarget(to)
                  return (
                    <Route
                      key={from}
                      path={from}
                      element={<Navigate to={{ pathname: path, search }} replace />}
                    />
                  )
                })}
                {/* The sentinel sits INSIDE each route's Suspense boundary,
                    beside the (lazy) page: React mounts a suspended boundary's
                    children only once its chunk has committed, so the
                    data-app-ready flag can never lead the route content it
                    declares readiness for. */}
                {discoveredRoutes.map(({ path, Component }) => (
                  <Route
                    key={path}
                    path={path}
                    element={
                      <Suspense fallback={<PageFallback />}>
                        <Component />
                        <AppReadySentinel />
                      </Suspense>
                    }
                  />
                ))}
                {/* Dev-only design lab — Three.js dock concept showcase.
                    Gated on DEV: the route disappears from production builds,
                    and lazy so three.js is never even fetched. */}
                {import.meta.env.DEV ? (
                  <Route
                    path="/design-lab"
                    element={
                      <Suspense fallback={<PageFallback />}>
                        <DesignLab3D />
                        <AppReadySentinel />
                      </Suspense>
                    }
                  />
                ) : null}
                {!hasHome ? <Route path="/" element={<Navigate to="/datasets" replace />} /> : null}
                {/* 404s render a real (static) route — the sentinel still has to
                    ride beside the content so readiness is declared for THIS
                    commit, not for the shell that loaded before it. */}
                <Route
                  path="*"
                  element={
                    <>
                      <NotFound />
                      <AppReadySentinel />
                    </>
                  }
                />
              </Route>
            </Routes>
          </BrowserRouter>
        </LocaleProvider>
      </ModelProvider>
    </ThemeProvider>
  )
}

const el = document.getElementById('root')
if (!el) throw new Error('#root missing')

createRoot(el).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
