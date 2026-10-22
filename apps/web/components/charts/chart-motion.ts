'use client'

import { useEffect, useState } from 'react'

/**
 * Design-system motion budget (DESIGN_SYSTEM.md ▸ Motion): micro-interactions
 * are duration-150; "no bouncing, no flashy effects". Recharts animates
 * internally (d3-ease + rAF), so CSS `motion-reduce:` alone cannot stop it —
 * series elements must receive `isAnimationActive`/`animationDuration` from
 * here (slice c, card 96d1507c).
 */
export const CHART_MOTION_MS = 150

const REDUCED_QUERY = '(prefers-reduced-motion: reduce)'

/**
 * Tracks the OS `prefers-reduced-motion` setting. Defaults to `false`
 * (animate) before mount and when `matchMedia` is unavailable (jsdom/SSR) —
 * animation is the visual default, the OS setting overrides it live.
 */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
    const mql = window.matchMedia(REDUCED_QUERY)
    setReduced(mql.matches)
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches)
    mql.addEventListener('change', onChange)
    return () => mql.removeEventListener('change', onChange)
  }, [])

  return reduced
}

/**
 * Props for any recharts series element (Line/Area/Bar/Radar): animation on
 * at the design-system duration, fully off for reduced-motion users.
 *
 *   <Line {...chartAnimation(reduced)} stroke={seriesColor(0)} … />
 */
export function chartAnimation(reduced: boolean) {
  return {
    isAnimationActive: !reduced,
    animationDuration: reduced ? 0 : CHART_MOTION_MS,
  }
}
