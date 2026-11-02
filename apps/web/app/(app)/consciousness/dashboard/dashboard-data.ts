// Shared data shapes, chart palettes, and formatters for the consciousness
// dashboard. Extracted from page.tsx by the page-decomposition refactor;
// everything here is pure — no React, no controllers.

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

export type BeliefsChartPoint = BeliefsPoint & { time: string }
export type QualiaChartPoint = QualiaPoint & { time: string }
export type GrowthChartPoint = { time: string; growth: number; cumulative: number }

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
