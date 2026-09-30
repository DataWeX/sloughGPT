import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

/**
 * Structured palette contract.
 *
 * `packages/strui/tokens/palette.json` is the single source of every color value in the
 * design system. Everything else — the two `globals.css` token regions, the web swatch
 * maps, the mobile color projection — is generated from it by
 * `packages/strui/scripts/gen-tokens.mjs` and verified here.
 *
 * Three invariants are guarded:
 *   1. sync      — every generated projection is byte-identical to a fresh render (`--check`)
 *   2. cascade   — tiers are ordered base -> palette -> aura, and auras may only ever set
 *                  `--primary`/`--ring` (the palette x aura cascade bug web already fixed)
 *   3. totality  — every palette defines every palette-scoped token in both modes, so no
 *                  token silently falls through to the base palette depending on mode
 */

const HERE = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(HERE, '../../../..')

const PALETTE = path.join(ROOT, 'packages/strui/tokens/palette.json')
const GEN = path.join(ROOT, 'packages/strui/scripts/gen-tokens.mjs')
const WEB_CSS = path.join(ROOT, 'apps/web/app/globals.css')
const STRUI_CSS = path.join(ROOT, 'packages/strui/src/styles/globals.css')
const WEB_SWATCH = path.join(ROOT, 'apps/web/lib/theme-tokens.generated.ts')
const MOBILE_PROJECTION = path.join(ROOT, 'apps/mobile/src/theme/palette.generated.ts')
const MOBILE_COLORS = path.join(ROOT, 'apps/mobile/src/theme/colors.ts')

interface Palette {
  palettes: Record<
    string,
    { label: string; base?: boolean; light: Record<string, string>; dark: Record<string, string> }
  >
  auras: Record<string, { label: string; note: string; primary: string; ring: string }>
  targets: Record<
    string,
    {
      css: string
      lightTokens: string[]
      darkTokens: string[]
      paletteTokens: string[]
      shadowTokens: string[]
    }
  >
}

const read = (p: string): string => readFileSync(p, 'utf8')
const loadPalette = (): Palette => JSON.parse(read(PALETTE))
const readCss = (p: string): string => read(p)

/** Content between a pair of `/* @tokens:NAME BEGIN\/END *\/` markers, or null. */
function region(css: string, name: string): string | null {
  const re = new RegExp(`/\\* @tokens:${name} BEGIN \\*/([\\s\\S]*?)/\\* @tokens:${name} END \\*/`)
  const m = css.match(re)
  return m ? m[1] : null
}

/** `--var: value;` declarations inside a block, in order. */
function declarations(block: string): Array<[string, string]> {
  return [...block.matchAll(/(--[\w-]+):\s*([^;]+);/g)].map((m) => [m[1], m[2].trim()])
}

/** Body of the first block whose selector list contains `selectorNeedle`. */
function blockFor(css: string, selectorNeedle: string): string | null {
  const idx = css.indexOf(selectorNeedle)
  if (idx === -1) return null
  const open = css.indexOf('{', idx)
  const close = css.indexOf('}', open)
  return open === -1 || close === -1 ? null : css.slice(open + 1, close)
}

/** `'124 82 196'` -> `'#7c52c4'` (lower) / `'#7C52C4'` (upper). */
function hex(rgb: string, case_: 'lower' | 'upper' = 'lower'): string {
  const [r, g, b] = rgb
    .trim()
    .split(/\s+/)
    .slice(0, 3)
    .map((n) => Number.parseInt(n, 10))
  const out = [r, g, b].map((n) => n.toString(16).padStart(2, '0')).join('')
  return case_ === 'lower' ? `#${out}` : `#${out.toUpperCase()}`
}

describe('palette source', () => {
  it('palette.json exists', () => {
    expect(existsSync(PALETTE), `missing ${PALETTE}`).toBe(true)
  })

  it('generator exists', () => {
    expect(existsSync(GEN), `missing ${GEN}`).toBe(true)
  })

  it('defines exactly two palettes and seven auras (the nine themes)', () => {
    const p = loadPalette()
    expect(Object.keys(p.palettes).sort()).toEqual(['neural-precision', 'noir-violet'])
    expect(Object.keys(p.auras).sort()).toEqual([
      'blue',
      'green',
      'orange',
      'pink',
      'purple',
      'red',
      'teal',
    ])
  })

  it('every palette defines identical token sets in light and dark (totality)', () => {
    const p = loadPalette()
    for (const [id, paletteDef] of Object.entries(p.palettes)) {
      expect(
        Object.keys(paletteDef.light).sort(),
        `palette ${id} light/dark token sets differ`,
      ).toEqual(Object.keys(paletteDef.dark).sort())
    }
  })

  it('every palette defines every palette-scoped token in both modes', () => {
    const p = loadPalette()
    const scoped = p.targets.web.paletteTokens
    for (const [id, paletteDef] of Object.entries(p.palettes)) {
      for (const mode of ['light', 'dark'] as const) {
        const missing = scoped.filter((t) => !(t in paletteDef[mode]))
        expect(missing, `palette ${id} (${mode}) missing ${missing.join(', ')}`).toEqual([])
      }
    }
  })

  it('auras expose exactly --primary and --ring', () => {
    const p = loadPalette()
    for (const [id, aura] of Object.entries(p.auras)) {
      expect(
        Object.keys(aura)
          .filter((k) => ['primary', 'ring'].includes(k))
          .sort(),
        `aura ${id}`,
      ).toEqual(['primary', 'ring'])
      expect(aura.primary).toMatch(/^\d+ \d+ \d+$/)
      expect(aura.ring).toMatch(/^\d+ \d+ \d+$/)
    }
  })

  it('target token inventories are covered by the base palette', () => {
    const p = loadPalette()
    const base = p.palettes['noir-violet']
    for (const [id, target] of Object.entries(p.targets)) {
      const missingLight = target.lightTokens.filter((t) => !(t in base.light))
      const missingDark = target.darkTokens.filter((t) => !(t in base.dark))
      expect(missingLight, `target ${id} light: ${missingLight.join(', ')}`).toEqual([])
      expect(missingDark, `target ${id} dark: ${missingDark.join(', ')}`).toEqual([])
    }
  })
})

describe('generated projections are in sync with palette.json', () => {
  it('gen-tokens.mjs --check passes', () => {
    let out = ''
    try {
      out = execFileSync(process.execPath, [GEN, '--check'], { encoding: 'utf8', cwd: ROOT })
    } catch (err) {
      const e = err as { stdout?: string; stderr?: string; message?: string }
      throw new Error(`--check failed:\n${e.stdout ?? ''}${e.stderr ?? ''}${e.message ?? ''}`)
    }
    expect(out).toContain('ok')
  })
})

describe('web globals.css cascade', () => {
  it('exposes all token regions', () => {
    const css = readCss(WEB_CSS)
    for (const name of ['light', 'dark', 'palettes', 'auras', 'shadows']) {
      expect(region(css, name), `web region @tokens:${name}`).not.toBeNull()
    }
  })

  it('orders tiers base -> palette -> aura', () => {
    const css = readCss(WEB_CSS)
    const light = css.indexOf('@tokens:light BEGIN')
    const palettes = css.indexOf('@tokens:palettes BEGIN')
    const auras = css.indexOf('@tokens:auras BEGIN')
    expect(light).toBeGreaterThan(-1)
    expect(palettes).toBeGreaterThan(light)
    expect(auras).toBeGreaterThan(palettes)
  })

  it('aura blocks set only --primary/--ring, identical across modes', () => {
    const css = readCss(WEB_CSS)
    const p = loadPalette()
    const auraRegion = region(css, 'auras')
    expect(auraRegion).not.toBeNull()
    for (const [id, aura] of Object.entries(p.auras)) {
      const block = blockFor(auraRegion!, `html.theme-${id},`)
      expect(block, `web aura block ${id}`).not.toBeNull()
      const decls = declarations(block!)
      expect(decls.map(([k]) => k).sort(), `web aura ${id} may only set primary+ring`).toEqual([
        '--primary',
        '--ring',
      ])
      expect(Object.fromEntries(decls)).toEqual({ '--primary': aura.primary, '--ring': aura.ring })
      // dark must share the same block so the accent survives html.dark
      expect(auraRegion).toContain(`html.dark.theme-${id} {`)
    }
  })
})

describe('strui globals.css mirrors the same doctrine', () => {
  it('exposes all token regions', () => {
    const css = readCss(STRUI_CSS)
    for (const name of ['light', 'dark', 'palettes', 'auras']) {
      expect(region(css, name), `strui region @tokens:${name}`).not.toBeNull()
    }
  })

  it('orders tiers base -> palette -> aura (palette must not overwrite the accent)', () => {
    const css = readCss(STRUI_CSS)
    const light = css.indexOf('@tokens:light BEGIN')
    const palettes = css.indexOf('@tokens:palettes BEGIN')
    const auras = css.indexOf('@tokens:auras BEGIN')
    expect(light).toBeGreaterThan(-1)
    expect(palettes).toBeGreaterThan(light)
    expect(auras).toBeGreaterThan(palettes)
  })

  it('aura blocks set only --primary/--ring — never surface tokens', () => {
    const css = readCss(STRUI_CSS)
    const p = loadPalette()
    const auraRegion = region(css, 'auras')
    expect(auraRegion).not.toBeNull()
    const surfaces = ['--secondary', '--secondary-foreground', '--muted', '--border', '--input']
    for (const [id, aura] of Object.entries(p.auras)) {
      const block = blockFor(auraRegion!, `html.theme-${id},`)
      expect(block, `strui aura block ${id}`).not.toBeNull()
      const keys = declarations(block!)
        .map(([k]) => k)
        .sort()
      expect(keys, `strui aura ${id}`).toEqual(['--primary', '--ring'])
      expect(keys.filter((k) => surfaces.includes(k))).toEqual([])
      expect(Object.fromEntries(declarations(block!))).toEqual({
        '--primary': aura.primary,
        '--ring': aura.ring,
      })
      expect(auraRegion).toContain(`html.dark.theme-${id} {`)
    }
  })
})

describe('web swatch projection', () => {
  it('is generated', () => {
    expect(existsSync(WEB_SWATCH), `missing ${WEB_SWATCH}`).toBe(true)
  })

  it('AURA_SWATCH / PALETTE_SWATCH equal the primary values in palette.json', async () => {
    const p = loadPalette()
    const mod = (await import(WEB_SWATCH)) as {
      AURA_SWATCH: Record<string, string>
      PALETTE_SWATCH: Record<string, string>
    }
    for (const [id, aura] of Object.entries(p.auras)) {
      expect(mod.AURA_SWATCH[id], `aura ${id} swatch`).toBe(hex(aura.primary))
    }
    for (const [id, def] of Object.entries(p.palettes)) {
      expect(mod.PALETTE_SWATCH[id], `palette ${id} swatch`).toBe(hex(def.light['--primary']))
    }
  })
})

describe('mobile projection', () => {
  it('is generated', () => {
    expect(existsSync(MOBILE_PROJECTION), `missing ${MOBILE_PROJECTION}`).toBe(true)
  })

  it('projects the web tokens (single source, both modes)', async () => {
    const p = loadPalette()
    const base = p.palettes['noir-violet']
    const mod = (await import(MOBILE_PROJECTION)) as {
      MOBILE_COLORS: Record<string, { light: string; dark: string }>
    }
    const mapping: Record<string, string> = {
      primary: '--primary',
      accent: '--accent',
      background: '--background',
      card: '--card',
      popover: '--popover',
      text: '--foreground',
      textMuted: '--muted-foreground',
      border: '--border',
      secondary: '--secondary',
      muted: '--muted',
      error: '--destructive',
      success: '--success',
      warning: '--warning',
      info: '--info',
      chatBg: '--chat-bg',
      textOnPrimary: '--primary-foreground',
      borderHover: '--primary',
    }
    for (const [key, token] of Object.entries(mapping)) {
      const got = mod.MOBILE_COLORS[key]
      expect(got, `mobile ${key}`).toBeDefined()
      expect(got.light, `mobile ${key} light`).toBe(hex(base.light[token], 'upper'))
      expect(got.dark, `mobile ${key} dark`).toBe(hex(base.dark[token], 'upper'))
    }
  })

  it('colors.ts carries no hardcoded hex literals', () => {
    const src = read(MOBILE_COLORS)
    const literals = src.match(/#[0-9a-fA-F]{6}\b/g) ?? []
    expect(literals, `hardcoded hex in colors.ts: ${literals.join(', ')}`).toEqual([])
  })

  it('no palette value is re-hardcoded outside the generated regions', () => {
    const p = loadPalette()

    // Every hex + RGB triple the palette owns, in every value form.
    const hexes = new Set<string>()
    const triples = new Set<string>()
    const collect = (value: string): void => {
      const m = value.match(/^(\d+) (\d+) (\d+)$/)
      if (m) {
        triples.add(value)
        hexes.add(hex(value))
      }
    }
    for (const def of Object.values(p.palettes)) {
      for (const mode of ['light', 'dark'] as const) Object.values(def[mode]).forEach(collect)
    }
    for (const aura of Object.values(p.auras)) {
      collect(aura.primary)
      collect(aura.ring)
    }

    const findStrays = (file: string): Record<string, number> => {
      // Drop the generated regions: everything else must not pin a palette hue.
      const stripped = read(file).replace(
        /\/\* @tokens:\w+ BEGIN \*\/[\s\S]*?\/\* @tokens:\w+ END \*\//g,
        '',
      )
      const strays: Record<string, number> = {}
      for (const value of [...hexes, ...triples]) {
        const re = new RegExp(
          `(?<![\\d.])${value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![\\d.])`,
          'gi',
        )
        const n = (stripped.match(re) ?? []).length
        if (n > 0) strays[value] = n
      }
      return strays
    }

    // strui's stylesheet must be 100% generated — zero strays.
    expect(findStrays(STRUI_CSS), 'strui globals.css pins palette values').toEqual({})

    // web's syntax-highlighter (.token.*) is a bespoke code theme that happens to
    // reuse two palette hues; it is out of the palette's scope but must not GROW.
    // If this fails, a new palette value was hardcoded outside @tokens — fix the
    // regression (or, if you are deliberately extending the code theme, say so here).
    expect(findStrays(WEB_CSS), 'web globals.css pins palette values').toEqual({
      '#7c52c4': 2,
      '#ec915f': 2,
    })
  })
})
