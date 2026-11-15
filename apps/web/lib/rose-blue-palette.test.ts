import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import {
  ROSE_BLUE_ID,
  ROSE_BLUE_LABEL,
  ROSE_BLUE_SWATCH,
  ROSE_BLUE_BASES,
  ROSE_BLUE_STATES,
  hexToRgb,
  rgbToHex,
  contrastRatio,
  deriveRamp,
  deriveTokens,
} from './rose-blue-palette'

const SHADES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900] as const

describe('rose-blue-palette protocol — registry constants', () => {
  it('exposes the id, label, and swatch the registry projects', () => {
    expect(ROSE_BLUE_ID).toBe('rose-blue')
    expect(ROSE_BLUE_LABEL).toBe('Rose Blue')
    expect(ROSE_BLUE_SWATCH).toBe('#5a5fc0')
  })

  it('bases are isolated constants (the provisional hue the user will retune)', () => {
    expect(ROSE_BLUE_BASES.light).toBe('#5a5fc0')
    expect(ROSE_BLUE_BASES.dark).toBe('#a8b6ff')
    expect(ROSE_BLUE_SWATCH).toBe(ROSE_BLUE_BASES.light)
  })
})

describe('rose-blue-palette protocol — color math', () => {
  it('hex and rgb convert exactly both ways', () => {
    expect(hexToRgb('#5a5fc0')).toEqual([90, 95, 192])
    expect(rgbToHex([90, 95, 192])).toBe('#5a5fc0')
    expect(rgbToHex(hexToRgb(ROSE_BLUE_BASES.dark))).toBe(ROSE_BLUE_BASES.dark)
  })
})

describe('rose-blue-palette protocol — state contract', () => {
  it('light states are the documented shade set', () => {
    expect(ROSE_BLUE_STATES.light).toEqual({
      selected: 700,
      hover: 800,
      active: 900,
      selectHoverBorder: 500,
    })
  })

  it('dark states are the documented shade set', () => {
    expect(ROSE_BLUE_STATES.dark).toEqual({
      selected: 800,
      hover: 900,
      active: 700,
      selectHoverBorder: 600,
    })
  })
})

describe('rose-blue-palette protocol — ramp invariants', () => {
  it.each(['light', 'dark'] as const)('%s ramp covers every shade at full opacity', (mode) => {
    const { shades, anchor } = deriveRamp(mode)
    for (const shade of SHADES) {
      const rgb = shades[shade]
      expect(rgb, `${mode}-${shade}`).toBeDefined()
      for (const c of rgb) {
        expect(c).toBeGreaterThanOrEqual(0)
        expect(c).toBeLessThanOrEqual(255)
        expect(Number.isInteger(c)).toBe(true)
      }
    }
    expect(SHADES).toContain(anchor)
  })

  it.each(['light', 'dark'] as const)('%s anchor shade snaps to the exact base color', (mode) => {
    const { shades, anchor } = deriveRamp(mode)
    expect(rgbToHex(shades[anchor])).toBe(ROSE_BLUE_BASES[mode])
  })

  it('light ramp gets darker as the shade number rises (monotonic in L)', () => {
    const rgbToL = (rgb: [number, number, number]) => {
      const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4))
      const [R, G, B] = rgb.map((c) => lin(c / 255)) as [number, number, number]
      const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B)
      const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B)
      const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B)
      return 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s
    }
    const { shades } = deriveRamp('light')
    let prev = Infinity
    for (const shade of SHADES) {
      const L = rgbToL(shades[shade])
      expect(L, `light-${shade} darker than the shade above`).toBeLessThan(prev)
      prev = L
    }
  })
})

describe('rose-blue-palette protocol — token projection', () => {
  it.each(['light', 'dark'] as const)('%s tokens cover the full palette block', (mode) => {
    const t = deriveTokens(mode)
    for (const key of [
      'background',
      'foreground',
      'card',
      'card-foreground',
      'popover',
      'popover-foreground',
      'primary',
      'primary-foreground',
      'secondary',
      'secondary-foreground',
      'muted',
      'muted-foreground',
      'accent',
      'accent-foreground',
      'ring',
      'border',
      'input',
      'chart-1',
      'chart-2',
      'chart-3',
      'chart-4',
      'chart-5',
    ])
      expect(t[key], key).toMatch(/^[\d ]+$/) // rgb triplets
    for (const key of ['shadow-sm', 'shadow-md', 'shadow-lg', 'shadow-xl'])
      expect(t[key], key).toMatch(/^0 [\d]+px [\d]+px rgba\(/)
    if (mode === 'dark') expect(t['chat-bg']).toMatch(/^[\d ]+$/)
  })

  it('primary projects to the base color, ring follows primary, input follows border', () => {
    const light = deriveTokens('light')
    expect(light['primary']).toBe('90 95 192')
    expect(light['ring']).toBe(light['primary'])
    expect(light['input']).toBe(light['border'])
    const dark = deriveTokens('dark')
    expect(dark['primary']).toBe('168 182 255')
    expect(dark['ring']).toBe(dark['primary'])
  })
})

describe('rose-blue-palette protocol — accessibility invariants', () => {
  it('white on light primary clears 4.5:1 (WCAG AA text)', () => {
    expect(contrastRatio([255, 255, 255], hexToRgb(ROSE_BLUE_BASES.light))).toBeGreaterThanOrEqual(
      4.5,
    )
  })

  it('dark primary foreground on dark primary clears 4.5:1', () => {
    const dark = deriveTokens('dark')
    const fg = dark['primary-foreground'].split(' ').map(Number) as [number, number, number]
    expect(contrastRatio(fg, hexToRgb(ROSE_BLUE_BASES.dark))).toBeGreaterThanOrEqual(4.5)
  })
})

describe('rose-blue-palette protocol — CSS projection drift guard', () => {
  const cssPath = fileURLToPath(new URL('../app/globals.css', import.meta.url))
  const css = readFileSync(cssPath, 'utf8')

  const parseBlock = (selector: string): Record<string, string> => {
    const start = css.indexOf(selector)
    expect(start, `selector ${selector} present in globals.css`).toBeGreaterThan(-1)
    const open = css.indexOf('{', start)
    const end = css.indexOf('}', open)
    const body = css.slice(open + 1, end)
    const out: Record<string, string> = {}
    for (const m of body.matchAll(/--([\w-]+):\s*([^;]+);/g)) out[m[1]] = m[2].trim()
    return out
  }

  it('html.palette-rose-blue block is byte-identical to the module derivation', () => {
    expect(parseBlock('html.palette-rose-blue {')).toEqual(deriveTokens('light'))
  })

  it('html.dark.palette-rose-blue block is byte-identical to the module derivation', () => {
    expect(parseBlock('html.dark.palette-rose-blue {')).toEqual(deriveTokens('dark'))
  })
})
