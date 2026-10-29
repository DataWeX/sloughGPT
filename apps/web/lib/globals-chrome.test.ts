import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

/**
 * Chrome-token contracts fixed by card 73ad1c78 (tooltip/badge audit):
 *
 * 1. Status-bar badges must be legible — 10px base (was 9px) and
 *    `.sl-badge-primary` must not be dimmed by `opacity` (it silently broke
 *    contrast on every primary badge).
 * 2. Every dark aura must override `--secondary-foreground` together with
 *    `--secondary`. The light blocks pair a pale `--secondary` with a deep
 *    foreground; the dark blocks swap to a deep `--secondary` but historically
 *    kept the deep foreground → ~1:1 contrast, invisible text on every
 *    `bg-secondary` element (selected cards, default badges) in dark mode.
 *    The dark foreground must be the aura's *pale* hue twin and pass 4.5:1.
 */

const css = readFileSync(fileURLToPath(new URL('../app/globals.css', import.meta.url)), 'utf8')

/** First `selector { ... }` block containing the given text. */
function blockWith(selector: string): string {
  const start = css.indexOf(`${selector} {`)
  expect(start, `selector ${selector} present in globals.css`).toBeGreaterThanOrEqual(0)
  return css.slice(start, css.indexOf('}', start))
}

/**
 * The LAST `selector { ... }` block — dark overrides appear after the shared
 * light+dark selector list, so the last match is the override block.
 */
function lastBlock(selector: string): string {
  let idx = -1
  let found = ''
  while ((idx = css.indexOf(`${selector} {`, idx + 1)) !== -1) {
    found = css.slice(idx, css.indexOf('}', idx))
  }
  expect(found, `dark override ${selector} present in globals.css`).not.toBe('')
  return found
}

function token(block: string, name: string): [number, number, number] {
  const m = block.match(new RegExp(`${name}:\\s*(\\d+)\\s+(\\d+)\\s+(\\d+)`))
  expect(m, `${name} defined in block`).not.toBeNull()
  return [Number(m![1]), Number(m![2]), Number(m![3])]
}

function channel(v: number): number {
  const c = v / 255
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

function luminance([r, g, b]: [number, number, number]): number {
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
}

function contrast(a: [number, number, number], b: [number, number, number]): number {
  const la = luminance(a) + 0.05
  const lb = luminance(b) + 0.05
  return Math.max(la, lb) / Math.min(la, lb)
}

const AURAS = ['blue', 'purple', 'pink', 'red', 'orange', 'green', 'teal'] as const

describe('status-bar badge chrome (card 73ad1c78)', () => {
  it('.sl-badge base font is 10px (9px was borderline illegible)', () => {
    const badge = blockWith('.sl-badge')
    expect(badge).toMatch(/font-size:\s*10px/)
    expect(badge).not.toMatch(/font-size:\s*9px/)
  })

  it('.sl-badge-primary is not dimmed by opacity', () => {
    expect(blockWith('.sl-badge-primary')).not.toMatch(/opacity/)
    expect(blockWith('.sl-badge-primary:hover')).not.toMatch(/opacity/)
  })
})

describe('dark aura secondary pairings (card 73ad1c78)', () => {
  it('every dark aura overrides --secondary-foreground alongside --secondary', () => {
    for (const aura of AURAS) {
      const override = lastBlock(`html.dark.theme-${aura}`)
      expect(override, `dark ${aura} sets --secondary-foreground`).toContain(
        '--secondary-foreground',
      )
      expect(override, `dark ${aura} sets --secondary`).toContain('--secondary:')
    }
  })

  it('dark secondary foreground passes WCAG 4.5:1 against its dark secondary', () => {
    for (const aura of AURAS) {
      const override = lastBlock(`html.dark.theme-${aura}`)
      const bg = token(override, '--secondary')
      const fg = token(override, '--secondary-foreground')
      const ratio = contrast(fg, bg)
      expect(ratio, `dark ${aura}: fg rgb(${fg}) on bg rgb(${bg})`).toBeGreaterThanOrEqual(4.5)
    }
  })
})
