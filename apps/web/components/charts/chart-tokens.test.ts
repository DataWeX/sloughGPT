import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { SERIES, AXIS, GRID, seriesColor } from './chart-tokens'

// Same resolution pattern as lib/page-journey-gate.test.ts.
const CHARTS_DIR = dirname(fileURLToPath(import.meta.url))
const source = (name: string) => readFileSync(join(CHARTS_DIR, name), 'utf8')

describe('chart-tokens', () => {
  it('series are rgb(var(--chart-N)) token references, never hex', () => {
    expect(SERIES).toHaveLength(5)
    for (const color of SERIES) {
      expect(color).toMatch(/^rgb\(var\(--chart-[1-5]\)\)$/)
    }
  })

  it('axis/grid are token references', () => {
    expect(AXIS).toBe('rgb(var(--muted-foreground))')
    expect(GRID).toBe('rgb(var(--border))')
  })

  it('seriesColor wraps around the palette', () => {
    expect(seriesColor(0)).toBe(SERIES[0])
    expect(seriesColor(4)).toBe(SERIES[4])
    expect(seriesColor(5)).toBe(SERIES[0])
    expect(seriesColor(7)).toBe(SERIES[2])
  })
})

describe('chart layer source: no hex color literals (DESIGN_SYSTEM.md chart rule)', () => {
  const files = ['ChartFrame.tsx', 'ChartTooltip.tsx', 'chart-tokens.ts']
  const lossChart = '../training/LossChart.tsx'

  it.each(files)('%s has no hex color', name => {
    expect(source(name)).not.toMatch(/#[0-9a-fA-F]{3,8}\b/)
  })

  it('%s has no hex color', () => {
    expect(source(lossChart)).not.toMatch(/#[0-9a-fA-F]{3,8}\b/)
  })
})
