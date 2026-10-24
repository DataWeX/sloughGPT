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
 * - personality/page: inline SVG stroke attributes
 * - planner/CardItem + planner/helpers: priority colours + card palettes
 * - chat useExportTemplates/useSessionGroups: export/group colour data
 * Fix the file, delete the row.
 */
const HEX_BASELINE = [
  'app/(app)/personality/page.tsx',
  'app/api/planner/helpers.ts',
  'components/planner/CardItem.tsx',
  'components/ThemeProvider.tsx',
  'features/chat/components/useExportTemplates.ts',
  'features/chat/components/useSessionGroups.ts',
]

/**
 * Pages still missing the `sl-page` wrapper (110 of 111 at baseline —
 * only magazine/page.tsx complies today). Adoption may only grow:
 * the count of non-compliant pages must never exceed this number.
 */
const SL_PAGE_MISSING_CEILING = 110

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

  it('sl-page wrapper adoption may only grow', () => {
    const missing = pages.filter(
      (p) => !readFileSync(join(WEB_ROOT, p), 'utf-8').includes('sl-page'),
    )
    expect(
      missing.length,
      `Pages missing the sl-page wrapper grew past the baseline ceiling. ` +
        `New pages MUST use: <div className="sl-page mx-auto max-w-4xl"> (designer.md).`,
    ).toBeLessThanOrEqual(SL_PAGE_MISSING_CEILING)
  })

  it('sanity: the catalogue of pages exists (fixture guard)', () => {
    expect(pages.length).toBeGreaterThan(50)
  })
})
