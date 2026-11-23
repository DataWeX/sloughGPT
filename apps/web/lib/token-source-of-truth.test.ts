import { describe, it, expect } from 'vitest'
import { execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

/**
 * `packages/strui/tokens/palette.json` is the single source of every colour
 * value in the web stylesheet; `gen-tokens.mjs` projects it into the
 * `@tokens:*` regions of `apps/web/app/globals.css`.
 *
 * A source of truth with no gate is only a hope — anyone hand-editing a
 * generated region (or palette.json without regenerating) must fail here
 * rather than ship a silent divergence.
 */

const WEB_DIR = fileURLToPath(new URL('..', import.meta.url))
const ROOT = path.resolve(WEB_DIR, '../..')
const GEN = path.join(ROOT, 'packages/strui/scripts/gen-tokens.mjs')
const CSS = path.join(WEB_DIR, 'app/globals.css')
const REGIONS = ['light', 'dark', 'palettes', 'auras', 'shadows'] as const

describe('token source of truth', () => {
  it('globals.css matches palette.json (generator --check)', () => {
    const out = execFileSync(process.execPath, [GEN, '--only=web', '--check'], {
      cwd: ROOT,
      encoding: 'utf8',
    })
    expect(out).toContain('ok — all generated artifacts match palette.json')
  })

  it('carries exactly one balanced marker pair per region', () => {
    const css = readFileSync(CSS, 'utf8')
    for (const region of REGIONS) {
      const begin = `/* @tokens:${region} BEGIN */`
      const end = `/* @tokens:${region} END */`
      expect(css.split(begin).length - 1, `${region} BEGIN`).toBe(1)
      expect(css.split(end).length - 1, `${region} END`).toBe(1)
      expect(css.indexOf(begin), `${region} order`).toBeLessThan(css.indexOf(end))
    }
  })

  it('keeps hand-owned palette blocks outside the generated region', () => {
    // system-cobalt / rose-blue are projected from lib/*-palette.ts and are
    // drift-guarded by their own tests; they must never be inside @tokens:palettes.
    const css = readFileSync(CSS, 'utf8')
    const begin = css.indexOf('/* @tokens:palettes BEGIN */')
    const end = css.indexOf('/* @tokens:palettes END */')
    const region = css.slice(begin, end)
    expect(region).toContain('html.palette-neural-precision')
    expect(region).not.toContain('palette-system-cobalt')
    expect(region).not.toContain('palette-rose-blue')
    expect(css.slice(end)).toContain('html.palette-system-cobalt')
    expect(css.slice(end)).toContain('html.palette-rose-blue')
  })

  it('keeps the high-contrast block last so it outranks html.dark.theme-*', () => {
    const css = readFileSync(CSS, 'utf8')
    const hc = css.indexOf('html.dark.high-contrast {')
    expect(hc).toBeGreaterThan(-1)

    // Walk to the block's closing brace so its own declarations don't count.
    let depth = 0
    let i = css.indexOf('{', hc)
    for (; i < css.length; i++) {
      if (css[i] === '{') depth++
      else if (css[i] === '}') {
        depth--
        if (depth === 0) break
      }
    }
    expect(depth, 'HC block closed').toBe(0)

    // The swap-transition rule sits after it but declares only `transition`,
    // never a colour, so it cannot affect HC's colour precedence.
    const after = css.slice(i)
    expect(/^\s*--[\w-]+\s*:/m.test(after), 'no colour token after the HC block').toBe(false)
    expect(after).toContain('html.theme-transitioning')
  })
})
