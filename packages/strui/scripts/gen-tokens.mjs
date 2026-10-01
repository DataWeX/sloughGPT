#!/usr/bin/env node
/**
 * Design-token generator for the Noir Violet design system.
 *
 *   node packages/strui/scripts/gen-tokens.mjs           # write generated artifacts
 *   node packages/strui/scripts/gen-tokens.mjs --check   # verify only; exit 1 on drift
 *
 * packages/strui/tokens/palette.json is the SINGLE SOURCE of every color value.
 * This script projects it into:
 *
 *   1. apps/web/app/globals.css                — the @tokens:* marker regions
 *   2. packages/strui/src/styles/globals.css   — the @tokens:* marker regions (storybook)
 *   3. apps/web/lib/theme-tokens.generated.ts  — swatch hex for the theme switcher
 *   4. apps/mobile/src/theme/palette.generated.ts — hex projection for React Native
 *   5. apps/mobile/src/theme/tamagui-themes.generated.ts — tamagui theme overrides
 *      (light/dark resolve from the mobile projection; accent themes are hand-authored)
 *   6. apps/web/lib/palette-showcase.generated.ts — RGB-triple reference for the
 *      magazine showcase page (editorial labels stay hand-written in page.tsx)
 *
 * Invariants this generator exists to hold:
 *   - tier order (base -> palette -> aura) is fixed by marker placement, never moved;
 *   - an aura only ever renders --primary/--ring (palette x aura cascade bug);
 *   - every palette renders every palette-scoped token in both modes (totality), so no
 *     token silently falls through to the base palette depending on mode.
 *
 * Zero dependencies. Edit palette.json, not the generated outputs.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(HERE, '../../..')
const PALETTE_FILE = path.join(HERE, '../tokens/palette.json')

/** Presentation config: which regions each stylesheet owns, and how they read. */
const TARGET_CONFIG = {
  web: { regions: ['light', 'dark', 'palettes', 'auras', 'shadows'], lightComment: true },
  strui: { regions: ['light', 'dark', 'palettes', 'auras'], lightComment: false },
}

const palette = JSON.parse(readFileSync(PALETTE_FILE, 'utf8'))
const comments = palette._comments
const base = palette.palettes['noir-violet']

const rel = (p) => path.relative(ROOT, p)
const fromRoot = (p) => path.join(ROOT, p)

/** `'124 82 196'` -> `'#7c52c4'` / `'#7C52C4'`. */
function hex(rgb, upper = false) {
  const [r, g, b] = rgb
    .trim()
    .split(/\s+/)
    .map((n) => Number.parseInt(n, 10))
  const out = [r, g, b].map((n) => n.toString(16).padStart(2, '0')).join('')
  return `#${upper ? out.toUpperCase() : out}`
}

function tokenLines(tokens, mode, source) {
  return tokens.map((t) => {
    const value = source[mode][t]
    if (value === undefined) {
      throw new Error(`palette source missing ${t} for mode "${mode}"`)
    }
    return `  ${t}: ${value};`
  })
}

// ---------------------------------------------------------------- region renders
function renderLight(targetId) {
  const target = palette.targets[targetId]
  const lines = []
  if (TARGET_CONFIG[targetId].lightComment) lines.push(comments.light)
  lines.push(...tokenLines(target.lightTokens, 'light', base))
  return lines.join('\n')
}

function renderDark(targetId) {
  return tokenLines(palette.targets[targetId].darkTokens, 'dark', base).join('\n')
}

function renderPalettes(targetId) {
  const target = palette.targets[targetId]
  const groups = []
  for (const [id, def] of Object.entries(palette.palettes)) {
    if (def.base) continue
    const lines = [comments.palettes]
    lines.push(`html.palette-${id} {`)
    lines.push(...tokenLines(target.paletteTokens, 'light', def))
    lines.push('}')
    lines.push(`html.dark.palette-${id} {`)
    lines.push(...tokenLines(target.paletteTokens, 'dark', def))
    lines.push('}')
    groups.push(lines.join('\n'))
  }
  return groups.join('\n\n')
}

function auraComment(aura) {
  const prefix = `/* ── Aura: ${aura.cssLabel} — ${aura.note}`
  const width = comments._auraLineWidth
  const pad = width - prefix.length - 4 // 1 leading space + 3 for " */"
  if (pad < 1) throw new Error(`aura comment too long to format: ${prefix}`)
  const line = `${prefix} ${'─'.repeat(pad)} */`
  if (line.length !== width)
    throw new Error(`aura comment width drift: ${line.length} !== ${width}`)
  return line
}

function renderAuras() {
  const blocks = Object.entries(palette.auras).map(([id, aura]) =>
    [
      auraComment(aura),
      `html.theme-${id},`,
      `html.dark.theme-${id} {`,
      `  --primary: ${aura.primary};`,
      `  --ring: ${aura.ring};`,
      '}',
    ].join('\n'),
  )
  return `${comments.doctrine}\n${blocks.join('\n\n')}`
}

function renderShadows(targetId) {
  const target = palette.targets[targetId]
  const lines = [comments.shadows, ':root {']
  lines.push(...tokenLines(target.shadowTokens, 'light', base))
  lines.push('}', 'html.dark {')
  lines.push(...tokenLines(target.shadowTokens, 'dark', base))
  lines.push('}')
  return lines.join('\n')
}

const RENDERERS = {
  light: renderLight,
  dark: renderDark,
  palettes: renderPalettes,
  auras: renderAuras,
  shadows: renderShadows,
}

// ------------------------------------------------------------------ css writing
function applyRegion(css, name, content) {
  const begin = `/* @tokens:${name} BEGIN */`
  const end = `/* @tokens:${name} END */`
  const i = css.indexOf(begin)
  const j = css.indexOf(end)
  if (i === -1 || j === -1) throw new Error(`missing @tokens:${name} markers`)
  if (j < i) throw new Error(`@tokens:${name} markers out of order`)
  return css.slice(0, i + begin.length) + '\n' + content + '\n' + css.slice(j)
}

function renderStylesheet(targetId) {
  const file = fromRoot(palette.targets[targetId].css)
  let css = readFileSync(file, 'utf8')
  for (const region of TARGET_CONFIG[targetId].regions) {
    css = applyRegion(css, region, RENDERERS[region](targetId))
  }
  return [file, css]
}

// ----------------------------------------------------------------- ts rendering
function renderSwatchTs() {
  const auras = Object.entries(palette.auras)
    .map(([id, a]) => `  ${id}: '${hex(a.primary)}',`)
    .join('\n')
  const palettes = Object.entries(palette.palettes)
    .map(([id, p]) => `  '${id}': '${hex(p.light['--primary'])}',`)
    .join('\n')
  return `/**
 * GENERATED FILE — do not edit. Source: packages/strui/tokens/palette.json
 * Regenerate: node packages/strui/scripts/gen-tokens.mjs
 */

/** Primary swatch per accent aura (ThemeSwitcher chips). */
export const AURA_SWATCH = {
${auras}
} as const

/** Primary swatch per full-spectrum palette (ThemeSwitcher chips). */
export const PALETTE_SWATCH = {
${palettes}
} as const
`
}

function renderMobileTs() {
  const rows = []
  for (const [key, ref] of Object.entries(palette.mobile.map)) {
    const lightToken = typeof ref === 'string' ? ref : ref.light
    const darkToken = typeof ref === 'string' ? ref : ref.dark
    rows.push(
      `  ${key}: { light: '${hex(base.light[lightToken], true)}', dark: '${hex(base.dark[darkToken], true)}' },`,
    )
  }
  for (const [key, value] of Object.entries(palette.mobile.extras)) {
    rows.push(`  ${key}: { light: '${value.light}', dark: '${value.dark}' },`)
  }
  return `/**
 * GENERATED FILE — do not edit. Source: packages/strui/tokens/palette.json
 * Regenerate: node packages/strui/scripts/gen-tokens.mjs
 *
 * Values are uppercase hex for React Native (matches the historic colors.ts contract).
 * Extra keys are mobile-only surfaces (hover/press states, tinted backgrounds) that the
 * web CSS derives with color-mix() instead of declaring as tokens.
 */
export const MOBILE_COLORS = {
${rows.join('\n')}
} as const
`
}

/** Resolve MOBILE_COLORS once (same rules as renderMobileTs). */
function mobileColors() {
  const colors = {}
  for (const [key, ref] of Object.entries(palette.mobile.map)) {
    const lightToken = typeof ref === 'string' ? ref : ref.light
    const darkToken = typeof ref === 'string' ? ref : ref.dark
    colors[key] = {
      light: hex(base.light[lightToken], true),
      dark: hex(base.dark[darkToken], true),
    }
  }
  for (const [key, value] of Object.entries(palette.mobile.extras)) {
    colors[key] = { light: value.light, dark: value.dark }
  }
  return colors
}

function renderTamaguiTs() {
  const t = palette.tamagui
  if (!t) throw new Error('palette.json missing the tamagui section')
  const colors = mobileColors()

  const modeBlock = (mode) =>
    Object.entries(t.map)
      .map(([field, ref]) => {
        const key = typeof ref === 'string' ? ref : ref[mode]
        const value = colors[key]?.[mode]
        if (value === undefined) {
          throw new Error(`tamagui ${mode}.${field}: unresolved mobile key "${key}"`)
        }
        return `    ${field}: '${value}',`
      })
      .join('\n')

  const blocks = [
    `  light: {\n${modeBlock('light')}\n  },`,
    `  dark: {\n${modeBlock('dark')}\n  },`,
    ...Object.entries(t.accents).map(([name, fields]) => {
      const lines = Object.entries(fields)
        .map(([field, value]) => `    ${field}: '${value}',`)
        .join('\n')
      return `  ${name}: {\n${lines}\n  },`
    }),
  ]

  return `/**
 * GENERATED FILE — do not edit. Source: packages/strui/tokens/palette.json
 * Regenerate: node packages/strui/scripts/gen-tokens.mjs
 *
 * tamagui theme overrides. light/dark resolve from the mobile projection (single
 * source with the web CSS); accent themes are hand-authored mobile-only hues.
 * tamagui.config.ts spreads these over the @tamagui/config defaults.
 */
export const TAMAGUI_THEMES = {
${blocks.join('\n')}
} as const
`
}

function renderShowcaseTs() {
  const s = palette.showcase
  if (!s) throw new Error('palette.json missing the showcase section')
  const def = palette.palettes[s.palette]
  if (!def) throw new Error(`showcase palette "${s.palette}" not found`)
  const rows = (mode) =>
    Object.entries(def[mode])
      .map(([token, rgb]) => `    '${token}': '${rgb}',`)
      .join('\n')
  const auraRows = Object.entries(palette.auras)
    .map(([id, aura]) => `    '${id}': '${aura.primary}',`)
    .join('\n')
  return `/**
 * GENERATED FILE — do not edit. Source: packages/strui/tokens/palette.json
 * Regenerate: node packages/strui/scripts/gen-tokens.mjs
 *
 * RGB-triple reference of the '${s.palette}' palette for the magazine showcase
 * page. Editorial labels/descriptions stay hand-written in page.tsx; every value
 * resolves from here so the showcase can never drift from the runtime theme.
 */
export const SHOWCASE_TOKENS = {
  light: {
${rows('light')}
  },
  dark: {
${rows('dark')}
  },
} as const

export const SHOWCASE_AURAS = {
${auraRows}
} as const
`
}

// ------------------------------------------------------------------------ main
const check = process.argv.includes('--check')

const outputs = [
  renderStylesheet('web'),
  renderStylesheet('strui'),
  [fromRoot(palette.swatch.generated), renderSwatchTs()],
  [fromRoot(palette.mobile.generated), renderMobileTs()],
  [fromRoot(palette.tamagui.generated), renderTamaguiTs()],
  [fromRoot(palette.showcase.generated), renderShowcaseTs()],
]

let drift = 0
for (const [file, content] of outputs) {
  let current = null
  try {
    current = readFileSync(file, 'utf8')
  } catch {
    current = null // missing generated file counts as drift in --check
  }
  if (current === content) {
    if (check) console.log(`ok       ${rel(file)}`)
    continue
  }
  if (check) {
    console.error(`DRIFT    ${rel(file)}`)
    drift += 1
  } else {
    writeFileSync(file, content)
    console.log(`wrote    ${rel(file)}`)
  }
}

if (check && drift > 0) {
  console.error(`${drift} generated artifact(s) out of sync with palette.json`)
  process.exit(1)
}
if (check) console.log('ok — all generated artifacts match palette.json')
