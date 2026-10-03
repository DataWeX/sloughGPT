/**
 * Vite entry — AppLayout shell + every app/(app) route via import.meta.glob.
 * Legacy REDIRECTS mount before discovered routes (redirects.ts parity).
 */
import './app/globals.css'

import { Suspense, StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Navigate, Outlet, Route, Routes } from 'react-router-dom'

import AppLayout from '@/components/AppLayout'
import AppNotFound from '@/app/not-found'
import { ThemeProvider } from '@/components/ThemeProvider'
import { ModelProvider } from '@/contexts/ModelContext'
import { LocaleProvider } from '@/hooks/useLocale'
import { ErrorLifecycle } from '@/components/ErrorLifecycle'
import { StartupOverlay } from '@/components/startup/StartupOverlay'
import { ConsciousnessQuickActionsWrapper } from '@/components/consciousness/ConsciousnessQuickActionsWrapper'
import { RouteDigestBoundary, ViteNavigatorBinder } from '@/vite/next-compat/digest-boundary'
import { discoveredRoutes } from '@/vite/routes'
import { REDIRECTS, splitTarget } from '@/lib/redirects'

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
      <ConsciousnessQuickActionsWrapper />
    </RouteDigestBoundary>
  )
}

function NotFound() {
  // Route-layer parity: reuse the real app/not-found.tsx instead of a divergent
  // placeholder (spec not-found-page expects its h1 + Home link).
  return <AppNotFound />
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
                {discoveredRoutes.map(({ path, Component }) => (
                  <Route
                    key={path}
                    path={path}
                    element={
                      <Suspense fallback={<PageFallback />}>
                        <Component />
                      </Suspense>
                    }
                  />
                ))}
                {!hasHome ? <Route path="/" element={<Navigate to="/datasets" replace />} /> : null}
                <Route path="*" element={<NotFound />} />
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
