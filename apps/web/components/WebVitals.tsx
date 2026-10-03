'use client'

/**
 * WebVitals — browser performance metrics collector (placeholder).
 *
 * In the Vite SPA, ``next/web-vitals`` resolves to the no-op compat shim
 * (vite/next-compat/web-vitals.ts) — its ``useReportWebVitals`` never invokes
 * the callback, so no CLS/INP/FCP/LCP/TTFB metrics are currently captured for
 * the web app. This mount point is kept so wiring the real ``web-vitals``
 * package later is a one-line import change; the callback below documents the
 * intended forwarding: warnings through the dev-log WebLogger, which in
 * production forwards warnings+ to ``POST /errors/logs/ingest`` and into the
 * server OutputBuffer, marking slow thresholds (LCP > 2500ms, INP > 200ms,
 * CLS > 0.1) in the ``slow`` context field.
 *
 * Rendering: ``null`` (instrumentation only).
 */

import { useReportWebVitals } from '@/vite/next-compat/web-vitals'
import { logger } from '@/lib/dev-log'

const _log = logger.child('web-vitals')

const SLOW_THRESHOLDS: Record<string, number> = {
  CLS: 0.1,
  INP: 200,
  FCP: 2000,
  LCP: 2500,
  TTFB: 800,
}

export default function WebVitals() {
  useReportWebVitals((metric) => {
    const { name, value, rating } = metric
    const threshold = SLOW_THRESHOLDS[name]
    const slow = threshold !== undefined && value > threshold
    _log.warning(`web-vitals ${name}`, {
      value: Math.round(value),
      rating,
      slow,
    })
  })
  return null
}
