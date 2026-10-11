/**
 * Shared shapes, chart palette and formatting for the Consciousness Dashboard.
 *
 * Extracted from the former 873-line page component so the section components
 * and the page share one definition instead of re-declaring them inline.
 */

import { formatTimeShort, toDateSeconds } from '@/lib/time-format'

export interface Episode {
  timestamp: number
  input_text: string
  self_insight: string
  growth_delta: number
  qualia: Record<string, number>
}

export interface QualiaPoint {
  timestamp: number
  valence: number
  arousal: number
  novelty: number
  coherence: number
}

export interface BeliefsPoint {
  timestamp: number
  step: number
  competence: number
  helpfulness: number
  creativity: number
  accuracy: number
  empathy: number
}

export interface EvalReport {
  overall_score: number
  diagnostics: string[]
  [key: string]: unknown
}

// Chart series routed through the theme tokens — chart-N follows the active
// theme/palette, so the dashboard never hardcodes a hue of its own.
export const BELIEF_COLORS: Record<string, string> = {
  competence: 'var(--chart-1)',
  helpfulness: 'var(--chart-2)',
  creativity: 'var(--chart-3)',
  accuracy: 'var(--chart-4)',
  empathy: 'var(--chart-5)',
}

export const QUALIA_COLORS: Record<string, string> = {
  valence: 'var(--chart-1)',
  arousal: 'var(--chart-5)',
  novelty: 'var(--chart-3)',
  coherence: 'var(--chart-2)',
}

export function formatTime(ts: number): string {
  return formatTimeShort(toDateSeconds(ts))
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

/** Chart series point for the beliefs line chart (adds a pre-formatted time label). */
export type BeliefsChartData = BeliefsPoint & { time: string }

/** Chart series point for the qualia area chart (adds a pre-formatted time label). */
export type QualiaChartData = QualiaPoint & { time: string }

/** Chart series point for the growth trajectory area chart. */
export interface GrowthPoint {
  time: string
  growth: number
  cumulative: number
}
