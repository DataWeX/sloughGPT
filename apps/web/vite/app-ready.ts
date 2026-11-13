/**
 * App-declared readiness signal — `data-app-ready` on the app root.
 *
 * The app, not the test, decides when it is interactive: the flag flips once
 * (a) the current route's content has committed and (b) startup reached the
 * stage the StartupOverlay itself treats as booted (`ready`/`background`).
 * Tests wait on this one predicate (avion `wait_app_ready`) instead of
 * guessing with `load`, `networkidle` (never settles — persistent SSE), root
 * child counts, or overlay absence, all of which measured as false signals.
 *
 * Set-once: the attribute survives SPA route navigations and is re-declared
 * after a full document load. Nothing here renders, so startup behaviour,
 * SSE wiring and the overlay are untouched.
 */

/** Attribute read by avion's APP_READY_EXPRESSION — keep the two in sync. */
export const APP_READY_ATTR = 'data-app-ready'

/** Stages at which the app considers itself booted (StartupOverlay.isReady). */
export function isStartupComplete(stage: string | null | undefined): boolean {
  return stage === 'ready' || stage === 'background'
}

/** Minimal DOM surface the flag is written to — faked in unit tests. */
export interface AppReadyTarget {
  hasAttribute(name: string): boolean
  setAttribute(name: string, value: string): void
}

/**
 * Declare readiness on *target*. Idempotent: the first caller wins and every
 * later call (another route mount, another effect pass) is a no-op.
 * Returns true only when this call actually set the attribute.
 */
export function markAppReady(target: AppReadyTarget | null | undefined): boolean {
  if (!target || target.hasAttribute(APP_READY_ATTR)) return false
  target.setAttribute(APP_READY_ATTR, 'true')
  return true
}
