/**
 * Shared formatters and chart-color tokens for the Consciousness Dashboard.
 *
 * These are module-level constants with no React or IO dependency, so they
 * live apart from the panels: `dashboard-evolution.tsx` and
 * `dashboard-state.tsx` both paint series with the same colors, and the
 * header, metrics and episode panels all format timestamps the same way.
 */

// Chart series routed through the theme tokens — chart-N follows the active
// theme/palette, so the dashboard never hardcodes a hue of its own.
export const BELIEF_COLORS: Record<string, string> = {
  competence: 'rgb(var(--chart-1))',
  helpfulness: 'rgb(var(--chart-2))',
  creativity: 'rgb(var(--chart-3))',
  accuracy: 'rgb(var(--chart-4))',
  empathy: 'rgb(var(--chart-5))',
}

export const QUALIA_COLORS: Record<string, string> = {
  valence: 'rgb(var(--chart-1))',
  arousal: 'rgb(var(--chart-5))',
  novelty: 'rgb(var(--chart-3))',
  coherence: 'rgb(var(--chart-2))',
}

export function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function formatTimeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

export function levelName(lvl: number): string {
  return lvl === 0 ? 'Off' : lvl === 1 ? 'Basic' : lvl === 2 ? 'Full' : 'Deep'
}
