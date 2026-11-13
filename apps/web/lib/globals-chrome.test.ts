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

/**
 * Banner tone surfaces (card 87345605):
 *
 * The banner used to be a single warning-tinted surface with 80%-opacity text
 * — tones were dead (all four rendered byte-identical) and light-mode text
 * computed to 1.44:1. The contract now passes WCAG 4.5:1 in both modes with
 * ONE formula, because the mode's own --foreground anchors the ink:
 *
 *   surface = --background (never tinted: a tone-tinted surface collapses
 *             contrast for dark tones — info-on-light lands at 1.9:1)
 *   ink     = 35% tone + 65% --foreground
 *   border  = 35% tone + 65% --background (decorative hue identity)
 *
 * 65% foreground (not less) is the binding number: light-mode warning is the
 * highest-luminance tone on the palest surface, and it still clears 7.7:1 —
 * margins that survive future palette drift.
 */
const TONE_TOKENS = {
  info: '--primary',
  success: '--success',
  warning: '--warning',
  destructive: '--destructive',
} as const

type RGB = [number, number, number]

function mix(a: RGB, b: RGB, t: number): RGB {
  return [a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t, a[2] * (1 - t) + b[2] * t]
}

describe('banner tone surfaces (card 87345605)', () => {
  it('.sl-app-banner derives every surface from tokens (no fixed colors, no opacity dimming)', () => {
    const block = blockWith('.sl-app-banner')
    // Tokens are RGB triplets — every reference MUST be wrapped rgb(), else
    // the declaration substitutes bare digits and invalidates (the live bug
    // this test now pins: transparent bg + inherited ink on every tone).
    expect(block).toMatch(/background:\s*rgb\(var\(--background\)\)/)
    expect(block).toMatch(
      /color:\s*color-mix\(in srgb,\s*rgb\(var\(--banner-tone\)\)\s*35%,\s*rgb\(var\(--foreground\)\)\)/,
    )
    expect(block).toMatch(
      /border(?:-bottom)?:\s*1px solid color-mix\(in srgb,\s*rgb\(var\(--banner-tone\)\)\s*35%,\s*rgb\(var\(--background\)\)\)/,
    )
    // Opacity-dimmed text silently breaks the ink contrast the next test pins.
    expect(block).not.toMatch(/opacity/)
    // No palette leaks: the base surface block names tokens, never colors.
    expect(block).not.toMatch(/rgb\(\s*\d/)
    expect(block).not.toMatch(/rgba\(/)
    expect(block).not.toMatch(/#[0-9a-fA-F]{3,8}\b/)
  })

  it('maps each data-tone to its semantic token', () => {
    for (const [tone, name] of Object.entries(TONE_TOKENS)) {
      const block = blockWith(`.sl-app-banner[data-tone='${tone}']`)
      expect(block, `${tone} maps to ${name}`).toContain(`--banner-tone: var(${name})`)
    }
  })

  it('has an entrance animation that reduced-motion users opt out of', () => {
    const block = blockWith('.sl-app-banner')
    expect(block).toMatch(/animation:\s*sl-banner-in/)
    // The animated properties live in the keyframes, never on the surface
    // block — the surface stays a pure token contract.
    const keyframes = css.slice(css.indexOf('@keyframes sl-banner-in'))
    expect(keyframes.slice(0, keyframes.indexOf('}'))).toMatch(/transform:/)
    // Reduced motion: same banner, no movement.
    const rmStart = css.indexOf('@media (prefers-reduced-motion: reduce) {\n  .sl-app-banner {')
    expect(rmStart, 'reduced-motion opt-out for the banner').toBeGreaterThanOrEqual(0)
    const rmBlock = css.slice(rmStart, css.indexOf('}', rmStart))
    expect(rmBlock).toMatch(/animation:\s*none/)
  })

  it('banner ink passes WCAG 4.5:1 in both modes for every tone', () => {
    for (const [mode, selector] of [
      ['light', ':root'],
      ['dark', 'html.dark'],
    ] as const) {
      const block = tokenBlock(selector, '--foreground:')
      const bg = token(block, '--background')
      const fg = token(block, '--foreground')
      for (const [tone, name] of Object.entries(TONE_TOKENS)) {
        const rgb = token(block, name)
        const ink = mix(rgb, fg, 0.65)
        const ratio = contrast(ink, bg)
        expect(
          ratio,
          `${mode}/${tone}: ink rgb(${ink.map(Math.round)}) on rgb(${bg})`,
        ).toBeGreaterThanOrEqual(4.5)
      }
    }
  })
})

/** The `${selector} {` block that actually declares the probe property —
 *  `:root` appears three times in globals.css (app-shell vars, color tokens,
 *  and a later utility block), so first-match is the wrong block. */
function tokenBlock(selector: string, probe: string): string {
  let idx = -1
  while ((idx = css.indexOf(`${selector} {`, idx + 1)) !== -1) {
    const block = css.slice(idx, css.indexOf('}', idx))
    if (block.includes(probe)) return block
  }
  throw new Error(`no ${selector} block containing ${probe} in globals.css`)
}
