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
 * - ThemeProvider: theme swatch *data* — the picker shows every theme's
 *   colour simultaneously while only one theme's tokens are live, so the
 *   swatch list cannot reference vars. Permanent data exemption.
 * Fix the file, delete the row. (Burned down 2026-10-24: CardItem priority
 * map + glow, material and personality recharts strokes, export templates
 * (live token snapshot), useSessionGroups colours themselves now
 * --group-N identity slots + its legacy hex migration retired (slots are
 * theme-constant and equal the old values), planner helpers (dead
 * COMMIT_COLORS deleted).)
 */
const HEX_BASELINE = [
  // Theme swatch *data* — same class as ThemeProvider below.
  'lib/os-blue-palette.ts',
  // theme-id → accent-hex map (picker data), same exemption.
  'lib/theme-storage.ts',
  'components/ThemeProvider.tsx',
]

/**
 * Numeric rgb()/rgba() literals NOT routed through a token (`rgb(var(--…))`).
 * Same contract as HEX_BASELINE: each row is debt, may only shrink.
 * - magazine: the page IS the styleguide — Section 7 renders the DARK MODE
 *   PREVIEW (pinned dark palette values are the content), and the swatch
 *   samples are literal palette entries. ThemeProvider-class data exemption.
 * Decoration (elevation `--elevation-rgb`, sheen `--sheen-rgb`) and drag-state
 * semantic tints moved onto tokens 2026-10-24; world material hues onto the
 * categorical `--group-N` slots; compare chart fills now match their own
 * stroke tokens.
 */
const RGB_BASELINE = ['app/(app)/magazine/page.tsx']

/**
 * Stock Tailwind palette utilities (the "AI-slop rainbow") — ZERO TOLERANCE
 * since 2026-10-24 (burned down 37 files / 156 instances -> 0 across four
 * rounds). Any hit in any scanned file fails: `STOCK_PALETTE_BASELINE` is
 * empty and `STOCK_PALETTE_TOTAL_CEILING = 0`. Fix = semantic utility
 * (`text-success`, `bg-destructive`, `text-primary`) or `var(--chart-N)` /
 * `var(--group-N)` for categorical identity. Adding a grandfather row here
 * is a regression, not an escape hatch — the total test pins the sum to 0.
 * See design-system-enforcement skill (§ Stock Tailwind Palette).
 */
const STOCK_PALETTE_RE =
  /\b(?:bg|text|border|ring|from|to|via|divide|decoration|accent|caret|fill|stroke|shadow|outline)-(?:indigo|fuchsia|cyan|sky|emerald|lime|amber|teal|rose|pink|red|yellow|green|blue|orange|purple)-(?:50|100|200|300|400|500|600|700|800|900|950)\b/g

/** Per-file grandfather list — empty: every stock-palette utility is now an offender. */
const STOCK_PALETTE_BASELINE: Record<string, number> = {}

/** Total instance ceiling — additions to the map cannot lift it. Zero tolerance. */
const STOCK_PALETTE_TOTAL_CEILING = 0

/** This rule also covers lib/ and contexts/ (class strings live there too). */
const STOCK_SCAN_ROOTS = [...SCAN_ROOTS, 'lib', 'contexts'] as const

/**
 * Neutral grey wheel utilities (gray/slate/zinc/neutral/stone). The Noir
 * Violet system replaces greys with semantic tokens (foreground,
 * muted-foreground, card, muted, border…). BASELINE RATCHET measured
 * 2026-10-29: 178 instances / 12 files — rows may only shrink; fixing a
 * file means deleting its row. `components/consciousness/*Notifications*`
 * and `*QuickActions*` are consciousness-page debt (page excluded from the
 * burn per product ruling) and may only shrink with the rest.
 */
const WHEEL_RE =
  /\b(?:bg|text|border|ring|from|to|via|divide|decoration|accent|caret|fill|stroke|shadow|outline)-(?:gray|slate|zinc|neutral|stone)-(?:50|100|200|300|400|500|600|700|800|900|950)\b/g

const WHEEL_BASELINE: Record<string, number> = {
  'components/consciousness/ConsciousnessNotificationsPanel.tsx': 15,
  'components/consciousness/ConsciousnessQuickActions.tsx': 15,
}

const WHEEL_TOTAL_CEILING = 30

/**
 * `text-lg` is not on the type scale (design-system-enforcement skill) —
 * nearest legal step is `text-base`. Baseline ratchet measured 2026-10-29:
 * 88 -> 27 instances; only consciousness pages remain (excluded from burn).
 */
const TEXT_LG_RE = /\btext-lg\b/g

const TEXT_LG_BASELINE: Record<string, number> = {
  'app/(app)/consciousness/benchmark/page.tsx': 8,
  'app/(app)/consciousness/test-runner/page.tsx': 5,
  'app/(app)/consciousness/playground/page.tsx': 4,
  'app/(app)/consciousness/health/page.tsx': 4,
  'app/(app)/consciousness/compare/page.tsx': 3,
  'app/(app)/consciousness/analytics/page.tsx': 1,
  'app/(app)/consciousness/docs/page.tsx': 1,
  'app/(app)/consciousness/master/page.tsx': 1,
}

const TEXT_LG_TOTAL_CEILING = 27

/**
 * `text-2xl` is reserved for `components/AppRouteHeader.tsx` (AppRouteHeader
 * page title). Baseline ratchet measured 2026-10-29: 41 instances / 17 files,
 * consciousness pages and the magazine styleguide page included as known
 * debt (magazine renders literal palette/type-scale samples — data class).
 */
const TEXT_2XL_RE = /\btext-2xl\b/g

const TEXT_2XL_BASELINE: Record<string, number> = {
  'app/(app)/consciousness/statistics/page.tsx': 6,
  'app/(app)/consciousness/analytics/page.tsx': 4,
  'app/(app)/consciousness/history/page.tsx': 4,
  'app/(app)/consciousness/master/page.tsx': 4,
  'components/phoneme/ProgressSummary.tsx': 4,
  'components/phoneme/PhonemeComparison.tsx': 3,
  'components/phoneme/ScoringCard.tsx': 3,
  'app/(app)/magazine/page.tsx': 2,
  'components/phoneme/MultipleChoiceQuiz.tsx': 2,
  'components/phoneme/ChallengeCard.tsx': 2,
  'app/(app)/consciousness/training/page.tsx': 1,
  'app/(app)/consciousness/insights/page.tsx': 1,
  'components/security/SecurityAuditCard.tsx': 1,
  'components/phoneme/PronunciationGame.tsx': 1,
  'components/phoneme/PronunciationRecording.tsx': 1,
  'components/phoneme/FlashcardCard.tsx': 1,
  'features/chat/components/dialogs/ReadFileSection.tsx': 1,
}

const TEXT_2XL_TOTAL_CEILING = 41

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

  it('no NEW file introduces stock Tailwind palette utilities', () => {
    const files = STOCK_SCAN_ROOTS.flatMap((root) => collectFiles(join(WEB_ROOT, root)))
    const offenders = files.filter((rel) => {
      const src = stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8'))
      return src.match(STOCK_PALETTE_RE)?.length
    })
    const unexpected = offenders.filter((f) => !(f in STOCK_PALETTE_BASELINE))
    expect(
      unexpected,
      `Stock Tailwind palette in NEW files — use text-success/bg-destructive/` +
        `text-primary or var(--chart-N) (design-system-enforcement skill). ` +
        `Offenders: ${unexpected.join(', ')}`,
    ).toEqual([])
  })

  it('stock palette per-file counts may only shrink', () => {
    for (const [rel, max] of Object.entries(STOCK_PALETTE_BASELINE)) {
      const full = join(WEB_ROOT, rel)
      const src = stripComments(readFileSync(full, 'utf-8'))
      const now = src.match(STOCK_PALETTE_RE)?.length ?? 0
      expect(
        now,
        `${rel}: stock-palette instances grew past baseline ${max} — fix the ` +
          `file, then lower the row.`,
      ).toBeLessThanOrEqual(max)
    }
  })

  it('stock palette TOTAL may only shrink (map additions cannot lift it)', () => {
    const files = STOCK_SCAN_ROOTS.flatMap((root) => collectFiles(join(WEB_ROOT, root)))
    let total = 0
    for (const rel of files) {
      const src = stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8'))
      total += src.match(STOCK_PALETTE_RE)?.length ?? 0
    }
    expect(
      total,
      `Stock-palette debt grew past ${STOCK_PALETTE_TOTAL_CEILING} instances. ` +
        `Fix files and lower the ceiling — never raise it.`,
    ).toBeLessThanOrEqual(STOCK_PALETTE_TOTAL_CEILING)
  })

  it('no NEW file introduces neutral-grey wheel utilities', () => {
    const offenders = STOCK_SCAN_ROOTS.flatMap((root) => collectFiles(join(WEB_ROOT, root))).filter(
      (rel) => stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8')).match(WHEEL_RE)?.length,
    )
    const unexpected = offenders.filter((f) => !(f in WHEEL_BASELINE))
    expect(
      unexpected,
      `Grey-wheel utilities in NEW files — use text-foreground / ` +
        `text-muted-foreground / bg-card / bg-muted / border-border ` +
        `(design-system-enforcement skill). Offenders: ${unexpected.join(', ')}`,
    ).toEqual([])
  })

  it('grey-wheel per-file counts may only shrink', () => {
    for (const [rel, max] of Object.entries(WHEEL_BASELINE)) {
      const src = stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8'))
      const now = src.match(WHEEL_RE)?.length ?? 0
      expect(
        now,
        `${rel}: grey-wheel instances grew past baseline ${max} — fix the ` +
          `file, then lower the row.`,
      ).toBeLessThanOrEqual(max)
    }
  })

  it('grey-wheel TOTAL may only shrink (map additions cannot lift it)', () => {
    const files = STOCK_SCAN_ROOTS.flatMap((root) => collectFiles(join(WEB_ROOT, root)))
    let total = 0
    for (const rel of files) {
      total +=
        stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8')).match(WHEEL_RE)?.length ?? 0
    }
    expect(
      total,
      `Grey-wheel debt grew past ${WHEEL_TOTAL_CEILING} instances. Fix files ` +
        `and lower the ceiling — never raise it.`,
    ).toBeLessThanOrEqual(WHEEL_TOTAL_CEILING)
  })

  it('no NEW file introduces out-of-scale text-lg', () => {
    const offenders = STOCK_SCAN_ROOTS.flatMap((root) => collectFiles(join(WEB_ROOT, root))).filter(
      (rel) => stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8')).match(TEXT_LG_RE)?.length,
    )
    const unexpected = offenders.filter((f) => !(f in TEXT_LG_BASELINE))
    expect(
      unexpected,
      `text-lg is not on the type scale — use text-base (design-system-` +
        `enforcement skill). Offenders: ${unexpected.join(', ')}`,
    ).toEqual([])
  })

  it('text-lg per-file counts may only shrink', () => {
    for (const [rel, max] of Object.entries(TEXT_LG_BASELINE)) {
      const src = stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8'))
      const now = src.match(TEXT_LG_RE)?.length ?? 0
      expect(now, `${rel}: text-lg grew past baseline ${max}`).toBeLessThanOrEqual(max)
    }
  })

  it('text-lg TOTAL may only shrink', () => {
    const files = STOCK_SCAN_ROOTS.flatMap((root) => collectFiles(join(WEB_ROOT, root)))
    let total = 0
    for (const rel of files) {
      total +=
        stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8')).match(TEXT_LG_RE)?.length ?? 0
    }
    expect(
      total,
      `text-lg debt grew past ${TEXT_LG_TOTAL_CEILING}. Fix files and lower ` +
        `the ceiling — never raise it.`,
    ).toBeLessThanOrEqual(TEXT_LG_TOTAL_CEILING)
  })

  it('no NEW file introduces out-of-scope text-2xl', () => {
    const offenders = STOCK_SCAN_ROOTS.flatMap((root) => collectFiles(join(WEB_ROOT, root))).filter(
      (rel) => stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8')).match(TEXT_2XL_RE)?.length,
    )
    const unexpected = offenders.filter((f) => !(f in TEXT_2XL_BASELINE))
    expect(
      unexpected,
      `text-2xl is reserved for AppRouteHeader — use the type scale ` +
        `(design-system-enforcement skill). Offenders: ${unexpected.join(', ')}`,
    ).toEqual([])
  })

  it('text-2xl per-file counts may only shrink', () => {
    for (const [rel, max] of Object.entries(TEXT_2XL_BASELINE)) {
      const src = stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8'))
      const now = src.match(TEXT_2XL_RE)?.length ?? 0
      expect(now, `${rel}: text-2xl grew past baseline ${max}`).toBeLessThanOrEqual(max)
    }
  })

  it('text-2xl TOTAL may only shrink', () => {
    const files = STOCK_SCAN_ROOTS.flatMap((root) => collectFiles(join(WEB_ROOT, root)))
    let total = 0
    for (const rel of files) {
      total +=
        stripComments(readFileSync(join(WEB_ROOT, rel), 'utf-8')).match(TEXT_2XL_RE)?.length ?? 0
    }
    expect(
      total,
      `text-2xl debt grew past ${TEXT_2XL_TOTAL_CEILING}. Fix files and lower ` +
        `the ceiling — never raise it.`,
    ).toBeLessThanOrEqual(TEXT_2XL_TOTAL_CEILING)
  })

  it('sanity: the catalogue of pages exists (fixture guard)', () => {
    expect(pages.length).toBeGreaterThan(50)
  })
})
