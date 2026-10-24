// @vitest-environment node
/**
 * Design criteria tests (AGENTS.md design criteria → test-writer contract).
 *
 * The `designer` agent owns the rules; this file encodes the two that are
 * source-scannable. Both are BASELINE RATCHETS (same shape as the Page Rule
 * gate): the lists may only shrink — fixing a violation means deleting its
 * baseline row. New violations fail the build.
 *
 * Rules live in:
 *   .opencode/skills/design-system-enforcement/SKILL.md  (tokens, no hex)
 *   .opencode/agents/designer.md                        (sl-page wrapper)
 */
import { describe, it, expect } from 'vitest'
import { readdirSync, readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const WEB_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')

const SCAN_ROOTS = ['app', 'components', 'features', 'hooks'] as const

/** 6-digit hex literals in source. */
const HEX_RE = /#[0-9a-fA-F]{6}\b/

/**
 * Strip comments so hex in prose (`// brand is #6366f1`) doesn't trip the
 * gate. `(^|[^:])\/\/` keeps URL schemes (`https://…`) intact; block
 * comments are removed first. Strings are not tokenised — acceptable: a hex
 * inside a string literal IS a colour the runtime can use.
 */
export function stripComments(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1')
}

/**
 * Files allowed to contain hex today (measured 2026-10-24). Each row is debt:
 * - ThemeProvider: theme swatch *data* (a colour picker shows colours)
 * - planner/helpers: server-side card colour palette (distinct hue per card)
 * - chat useSessionGroups: group colour data (distinct hue per group)
 * Fix the file, delete the row. (Burned down 2026-10-24: CardItem priority
 * map + material and personality recharts strokes; export templates now
 * snapshot live tokens into the standalone stylesheet via rgb(var(--…)).)
 */
const HEX_BASELINE = [
  'app/api/planner/helpers.ts',
  'components/ThemeProvider.tsx',
  'features/chat/components/useSessionGroups.ts',
]

/**
 * Numeric rgb()/rgba() literals NOT routed through a token (`rgb(var(--…))`).
 * Same contract as HEX_BASELINE: each row is debt, may only shrink.
 * - magazine: the page IS the brand gallery (18 literals, design-owned)
 * - world: 2 extra hues beyond the 5 chart slots in an inline palette
 * - compare: recharts area fills (violet/cyan alpha ramps)
 * - kanban: neutral elevation shadow
 * - CardItem/BuffetEngine/CardEditor/Tray: dark-canvas decoration —
 *   elevation shadows, hairline borders, scrims (no hue identity; the two
 *   drag-state hairlines in Tray are pre-token tints pending a design pass)
 */
const RGB_BASELINE = [
  'app/(app)/magazine/page.tsx',
  'app/(app)/world/page.tsx',
  'app/(app)/kanban/page.tsx',
  'app/(app)/consciousness/compare/page.tsx',
  'components/planner/BuffetEngine.tsx',
  'components/planner/CardEditor.tsx',
  'components/planner/CardItem.tsx',
  'components/planner/Tray.tsx',
]

/** Numeric rgb()/rgba() — `rgb(var(--…))` is tokenised and does not match. */
const RGB_RE = /\brgba?\(\s*\d/

/**
 * Full-bleed surfaces (measured 2026-10-24: 2). These re-export workspace
 * layouts where a centered max-w column is wrong — the wrapper belongs to
 * their own layout, not PageContainer. Exemptions are debt too: this list
 * may only shrink, and only by converting the surface.
 */
const FULL_BLEED = ['app/(app)/chat/page.tsx', 'app/(app)/planner/page.tsx']

/** Redirect-only routes render no DOM — the wrapper rule cannot apply. */
function isRedirectOnly(src: string): boolean {
  return (
    src.includes('redirect(') || (src.includes('router.replace(') && src.includes('return null'))
  )
}

function collectFiles(dir: string, out: string[] = []): string[] {
  let entries
  try {
    entries = readdirSync(dir, { withFileTypes: true })
  } catch {
    return out
  }
  for (const entry of entries) {
    if (entry.name === 'node_modules' || entry.name === '.next') continue
    const full = join(dir, entry.name)
    if (entry.isDirectory()) collectFiles(full, out)
    else if (/\.tsx?$/.test(entry.name) && !/\.test\.(ts|tsx)$/.test(entry.name)) {
      out.push(
        full
          .slice(WEB_ROOT.length + 1)
          .split('\\')
          .join('/'),
      )
    }
  }
  return out
}

const scanned = SCAN_ROOTS.flatMap((root) => collectFiles(join(WEB_ROOT, root)))

const pages = scanned.filter((p) => /\/page\.tsx$/.test(p))

describe('Design criteria — Noir Violet ratchets', () => {
  it('no NEW source file introduces hex colours', () => {
    const offenders = scanned.filter((rel) =>
      HEX_RE.test(stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8'))),
    )
    const unexpected = offenders.filter((f) => !HEX_BASELINE.includes(f))
    expect(
      unexpected,
      `New hex literals in these files — use rgb(var(--token)) instead ` +
        `(design-system-enforcement skill). Offenders: ${unexpected.join(', ')}`,
    ).toEqual([])
  })

  it('no NEW source file introduces raw rgb() colours', () => {
    const offenders = scanned.filter((rel) =>
      RGB_RE.test(stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8'))),
    )
    const unexpected = offenders.filter((f) => !RGB_BASELINE.includes(f))
    expect(
      unexpected,
      `Raw rgb() literals — use rgb(var(--token)) or a chart/semantic token ` +
        `(design-system-enforcement skill). Offenders: ${unexpected.join(', ')}`,
    ).toEqual([])
  })

  it('rgb baseline may only shrink', () => {
    const offenders = scanned.filter((rel) =>
      RGB_RE.test(stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8'))),
    )
    expect(
      offenders.filter((f) => RGB_BASELINE.includes(f)).length,
      `Raw-rgb debt grew past the baseline (${RGB_BASELINE.length} files). ` +
        `Fix the file, delete the row.`,
    ).toBeLessThanOrEqual(RGB_BASELINE.length)
  })

  it('hex baseline may only shrink', () => {
    const offenders = scanned.filter((rel) =>
      HEX_RE.test(stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8'))),
    )
    expect(
      offenders.length,
      `Hex debt grew past baseline (${HEX_BASELINE.length}). Delete a baseline ` +
        `row only by fixing the file, never by adding one.`,
    ).toBeLessThanOrEqual(HEX_BASELINE.length)
  })

  it('every rendered page uses the wrapper (redirects/full-bleed exempt)', () => {
    const missing = pages.filter((p) => {
      if (FULL_BLEED.includes(p)) return false
      const src = readFileSync(join(WEB_ROOT, p), 'utf-8')
      if (isRedirectOnly(src)) return false
      return !(src.includes('sl-page') || src.includes('PageContainer'))
    })
    expect(
      missing,
      `New pages must use PageContainer (renders .sl-page) or the literal ` +
        `sl-page wrapper — designer.md. Redirect routes and the documented ` +
        `FULL_BLEED list are the only exemptions. Offenders: ${missing.join(', ')}`,
    ).toEqual([])
  })

  it('full-bleed exemption list may only shrink (max 2 today)', () => {
    expect(FULL_BLEED.length).toBeLessThanOrEqual(2)
    for (const p of FULL_BLEED) {
      expect(pages, `FULL_BLEED entry no longer exists: ${p}`).toContain(p)
    }
  })

  it('stripComments: hex in comments is ignored, code and URLs survive', () => {
    const commented = `
      // brand: #aaaaaa
      /* another: #bbbbbb */
      const url = 'https://example.com/#abcdef'
      const brand = '#123456'
    `
    const stripped = stripComments(commented)
    expect(stripped).not.toContain('#aaaaaa')
    expect(stripped).not.toContain('#bbbbbb')
    expect(stripped).toContain('https://example.com/#abcdef')
    expect(stripped).toContain('#123456')
    expect(HEX_RE.test(stripped)).toBe(true)
    expect(HEX_RE.test(stripComments('// only #abcdef in a comment'))).toBe(false)
  })

  it('sanity: the catalogue of pages exists (fixture guard)', () => {
    expect(pages.length).toBeGreaterThan(50)
  })
})
