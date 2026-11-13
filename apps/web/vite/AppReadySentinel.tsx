/**
 * AppReadySentinel — sets `data-app-ready` on #root at the one moment the app
 * can prove it is interactive.
 *
 * Mount it as a SIBLING of the route's content, inside that route's own
 * <Suspense> boundary (see vite-entry.tsx). React does not mount a boundary's
 * children while it is suspended, so this component's first effect cannot run
 * before the route chunk has committed; the startup-stage gate cannot pass
 * before the app reports `ready`/`background`. The attribute therefore only
 * ever flips during an effect pass whose commit already contains the route's
 * own content — at the instant a test observes `data-app-ready`, the page's
 * inputs are in the DOM.
 *
 * Renders nothing: no layout, no startup behaviour, no SSE wiring changes.
 */
import { useEffect } from 'react'

import { useLiveStatusStore } from '@/hooks/useLiveStatus'

import { isStartupComplete, markAppReady } from './app-ready'

export function AppReadySentinel() {
  const stage = useLiveStatusStore((s) => s.health?.startup_stage ?? 'unknown')
  const ready = isStartupComplete(stage)

  useEffect(() => {
    if (!ready) return
    markAppReady(document.getElementById('root'))
  }, [ready])

  return null
}
