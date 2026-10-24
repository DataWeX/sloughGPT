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

/** 6-digit hex literals in source (comments excluded by the tokeniser-less scan). */
const HEX_RE = /#[0-9a-fA-F]{6}\b/

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
      HEX_RE.test(readFileSync(join(WEB_ROOT, rel), 'utf-8')),
    )
    const unexpected = offenders.filter((f) => !HEX_BASELINE.includes(f))
    expect(
      unexpected,
      `New hex literals in these files — use rgb(var(--token)) instead ` +
        `(design-system-enforcement skill). Offenders: ${unexpected.join(', ')}`,
    ).toEqual([])
  })

  it('hex baseline may only shrink', () => {
    const offenders = scanned.filter((rel) =>
      HEX_RE.test(readFileSync(join(WEB_ROOT, rel), 'utf-8')),
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

  it('sanity: the catalogue of pages exists (fixture guard)', () => {
    expect(pages.length).toBeGreaterThan(50)
  })
})
