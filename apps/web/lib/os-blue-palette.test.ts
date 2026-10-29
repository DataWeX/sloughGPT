import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import {
  OS_BLUE_ID,
  OS_BLUE_LABEL,
  OS_BLUE_SWATCH,
  OS_BLUE_BASES,
  OS_BLUE_STATES,
  hexToRgb,
  rgbToHex,
  contrastRatio,
  deriveRamp,
  deriveTokens,
} from './os-blue-palette'

const SHADES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900] as const

describe('os-blue-palette protocol — registry constants', () => {
  it('exposes the hybrid naming the user chose', () => {
    expect(OS_BLUE_ID).toBe('system-cobalt')
    expect(OS_BLUE_LABEL).toBe('Operating System Blue')
    expect(OS_BLUE_SWATCH).toBe('#17599d')
  })

  it('bases are the user-picked hovered color (light) and the Win11 accent (dark)', () => {
    expect(OS_BLUE_BASES.light).toBe('#17599d')
    expect(OS_BLUE_BASES.dark).toBe('#4cc2ff')
  })
})

describe('os-blue-palette protocol — color math', () => {
  it('hex and rgb convert exactly both ways', () => {
    expect(hexToRgb('#17599d')).toEqual([23, 89, 157])
    expect(rgbToHex([23, 89, 157])).toBe('#17599d')
    expect(rgbToHex(hexToRgb('#4cc2ff'))).toBe('#4cc2ff')
  })

  it('contrast ratio matches known values', () => {
    // sanity anchors: black/white is 21:1, identical colors are 1:1
    expect(contrastRatio([0, 0, 0], [255, 255, 255])).toBeCloseTo(21, 1)
    expect(contrastRatio([120, 120, 120], [120, 120, 120])).toBeCloseTo(1, 5)
  })
})

describe('os-blue-palette protocol — ramp derivation', () => {
  it.each(['light', 'dark'] as const)('%s ramp has all 10 shades', (mode) => {
    const ramp = deriveRamp(mode)
    for (const s of SHADES) expect(ramp.shades[s]).toBeDefined()
  })

  it('light anchor snaps to the exact base color #17599d', () => {
    const ramp = deriveRamp('light')
    expect(ramp.anchor).toBe(700)
    expect(ramp.shades[700]).toEqual([23, 89, 157])
  })

  it('dark anchor snaps to the exact base color #4cc2ff', () => {
    const ramp = deriveRamp('dark')
    expect(ramp.anchor).toBe(800)
    expect(ramp.shades[800]).toEqual([76, 194, 255])
  })

  it('keeps hue constant across every shade (the OKLCH invariant)', () => {
    for (const mode of ['light', 'dark'] as const) {
      const ramp = deriveRamp(mode)
      const hues = SHADES.map((s) => rgbToOklchHue(ramp.shades[s]))
      const spread = Math.max(...hues) - Math.min(...hues)
      expect(spread).toBeLessThan(0.08) // < ~4.6° of hue drift incl. gamut clamp
    }
  })

  it('is strictly ordered by lightness — light darkens with shade index, dark lightens', () => {
    const light = SHADES.map((s) => rgbToOklchLightness(deriveRamp('light').shades[s]))
    for (let i = 1; i < light.length; i++) expect(light[i]).toBeLessThan(light[i - 1])
    const dark = SHADES.map((s) => rgbToOklchLightness(deriveRamp('dark').shades[s]))
    for (let i = 1; i < dark.length; i++) expect(dark[i]).toBeGreaterThan(dark[i - 1])
  })
})

describe('os-blue-palette protocol — state contract', () => {
  it('selected IS the base color in both modes', () => {
    expect(deriveRamp('light').shades[OS_BLUE_STATES.light.selected]).toEqual([23, 89, 157])
    expect(deriveRamp('dark').shades[OS_BLUE_STATES.dark.selected]).toEqual([76, 194, 255])
  })

  it('light interaction darkens (hover then active), dark hover lightens and active presses darker', () => {
    expect(OS_BLUE_STATES.light).toEqual({
      selected: 700,
      hover: 800,
      active: 900,
      selectHoverBorder: 500,
    })
    expect(OS_BLUE_STATES.dark).toEqual({
      selected: 800,
      hover: 900,
      active: 700,
      selectHoverBorder: 600,
    })
  })
})

describe('os-blue-palette protocol — token projection', () => {
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
    expect(light['primary']).toBe('23 89 157')
    expect(light['ring']).toBe(light['primary'])
    expect(light['input']).toBe(light['border'])
    const dark = deriveTokens('dark')
    expect(dark['primary']).toBe('76 194 255')
    expect(dark['ring']).toBe(dark['primary'])
  })
})

describe('os-blue-palette protocol — accessibility invariants', () => {
  it('white on light primary clears 4.5:1 (WCAG AA text)', () => {
    expect(contrastRatio([255, 255, 255], hexToRgb('#17599d'))).toBeGreaterThanOrEqual(4.5)
  })

  it('dark primary foreground on dark primary clears 4.5:1', () => {
    const dark = deriveTokens('dark')
    const fg = dark['primary-foreground'].split(' ').map(Number) as [number, number, number]
    expect(contrastRatio(fg, hexToRgb('#4cc2ff'))).toBeGreaterThanOrEqual(4.5)
  })
})

describe('os-blue-palette protocol — CSS projection drift guard', () => {
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

  it('html.palette-system-cobalt block is byte-identical to the module derivation', () => {
    expect(parseBlock('html.palette-system-cobalt {')).toEqual(deriveTokens('light'))
  })

  it('html.dark.palette-system-cobalt block is byte-identical to the module derivation', () => {
    expect(parseBlock('html.dark.palette-system-cobalt {')).toEqual(deriveTokens('dark'))
  })
})

// ── local OKLCH readers (independent re-implementation: the invariant test
//    must not share the module's own conversion code it is validating) ──
function lin(c: number) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}
function rgbToOklch([r, g, b]: [number, number, number]) {
  const R = lin(r / 255),
    G = lin(g / 255),
    B = lin(b / 255)
  const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B)
  const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B)
  const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B)
  const L = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s
  const A = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s
  const Bb = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s
  return { L, C: Math.hypot(A, Bb), H: Math.atan2(Bb, A) }
}
function rgbToOklchHue(rgb: [number, number, number]) {
  return rgbToOklch(rgb).H
}
function rgbToOklchLightness(rgb: [number, number, number]) {
  return rgbToOklch(rgb).L
}
