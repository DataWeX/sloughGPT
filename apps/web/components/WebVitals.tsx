'use client'

/**
 * WebVitals — browser performance metrics collector.
 *
 * Uses Next.js built-in ``useReportWebVitals`` (bundled ``web-vitals`` 3.0.0,
 * no extra dependency) to capture CLS, INP, FCP, LCP, TTFB.
 *
 * Metrics are forwarded at ``warning`` level through the dev-log WebLogger,
 * which in production forwards warnings+ to ``POST /errors/logs/ingest`` and
 * into the server OutputBuffer — tailable via ``/system/output`` and
 * ``/system/stream``. Slow thresholds (LCP > 2500ms, INP > 200ms, CLS > 0.1)
 * are marked in the ``slow`` context field so a hang surfaces as a visible
 * spike in the server log stream.
 *
 * Rendering: ``null`` (instrumentation only).
 */

import { useReportWebVitals } from 'next/web-vitals'
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
