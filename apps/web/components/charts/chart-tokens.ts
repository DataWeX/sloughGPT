/**
 * Chart color tokens — DESIGN_SYSTEM.md chart rule: every series/swatch
 * references a token, never a hex literal (forbidden-pattern row; the
 * design-system grep `grep -rn "': '#" apps/web/components apps/web/app`
 * must stay zero). Values live in app/globals.css (--chart-1..5, per-aura
 * variants) and resolve at runtime.
 */

/** Series palette, in default order of use. */
export const SERIES = [
  'rgb(var(--chart-1))',
  'rgb(var(--chart-2))',
  'rgb(var(--chart-3))',
  'rgb(var(--chart-4))',
  'rgb(var(--chart-5))',
] as const

/** Wraps around the palette for estates with more than five series. */
export function seriesColor(index: number): string {
  return SERIES[index % SERIES.length]
}

/** Axis strokes, ticks and labels. */
export const AXIS = 'rgb(var(--muted-foreground))'

/** Cartesian grid lines. */
export const GRID = 'rgb(var(--border))'
