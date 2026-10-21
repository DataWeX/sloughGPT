// @vitest-environment node
/**
 * Page Rule gate (AGENTS.md — "Page Rule (UI surfaces are journey-projected)").
 *
 * A page is a projection of a user journey, not a trophy case of engines.
 * This test keeps the two ratchets honest:
 *
 *  - `page-baseline.json` must mirror the disk EXACTLY: every `page.tsx`
 *    under apps/web/app is catalogued, and every catalogue entry exists.
 *    Adding a page = adding an entry to `journeys` with a `ux/<flow>` id
 *    cited from docs/UX_FLOWS.md. New pages can NEVER enter
 *    `grandfathered` — the grandfathered count only ever shrinks.
 *  - Deleting a page must delete its entry (no stale catalogue rows), and
 *    consolidation work should decrement `GRANDFATHERED_CEILING` along
 *    with the list.
 */
import { describe, it, expect } from 'vitest'
import { readdirSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import baseline from './page-baseline.json'

const WEB_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const APP_ROOT = join(WEB_ROOT, 'app')

/**
 * Measured on main 2026-10-17 (AGENTS.md Page Rule cites the same numbers).
 * 2026-10-21: 110 -> 109 — the `/images` page was absorbed into `/files`
 * (route-map row `/files` <- `/files` + `/images`, card 7839a7c6), and the
 * old path now 307s through `lib/redirects.ts`.
 */
const GRANDFATHERED_CEILING = 109

interface PageBaseline {
  grandfathered: string[]
  journeys: { path: string; journeyId: string }[]
}

const catalog = baseline as PageBaseline

function collectPages(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name)
    if (entry.isDirectory()) collectPages(full, out)
    else if (entry.name === 'page.tsx') out.push(full)
  }
  return out
}

/** Paths relative to apps/web, forward slashes — the catalogue format. */
const diskPages = collectPages(APP_ROOT)
  .map((p) => p.slice(WEB_ROOT.length + 1).split('\\').join('/'))
  .sort()

describe('Page Rule — page baseline ratchet', () => {
  it('every page.tsx on disk is catalogued (no uncatalogued surfaces)', () => {
    const catalogued = new Set([...catalog.grandfathered, ...catalog.journeys.map((j) => j.path)])
    const uncatalogued = diskPages.filter((p) => !catalogued.has(p))
    expect(
      uncatalogued,
      `New page.tsx must be registered in apps/web/lib/page-baseline.json with a ux/ journey id cited from docs/UX_FLOWS.md — AGENTS.md "Page Rule". Offenders: ${uncatalogued.join(', ')}`,
    ).toEqual([])
  })

  it('every catalogue entry exists on disk (deleting a page deletes its row)', () => {
    const onDisk = new Set(diskPages)
    const stale = [...catalog.grandfathered, ...catalog.journeys.map((j) => j.path)].filter(
      (p) => !onDisk.has(p),
    )
    expect(
      stale,
      `Stale page-baseline.json entries for deleted pages (remove them, and decrement GRANDFATHERED_CEILING if a grandfathered page went away): ${stale.join(', ')}`,
    ).toEqual([])
  })

  it('grandfathered count never grows (ratchet)', () => {
    expect(catalog.grandfathered.length).toBeLessThanOrEqual(GRANDFATHERED_CEILING)
  })

  it('new pages cite a ux/ journey — they can never self-grandfather', () => {
    const grandfathered = new Set(catalog.grandfathered)
    for (const entry of catalog.journeys) {
      expect(entry.journeyId, `${entry.path} must cite a docs/UX_FLOWS.md flow as ux/<flow>`).toMatch(
        /^ux\/[a-z0-9][a-z0-9-]*$/,
      )
      expect(grandfathered.has(entry.path), `${entry.path} is in both lists`).toBe(false)
    }
  })

  it('catalogue has no duplicate rows', () => {
    const all = [...catalog.grandfathered, ...catalog.journeys.map((j) => j.path)]
    expect(new Set(all).size, 'duplicate paths in page-baseline.json').toBe(all.length)
  })

  it('the pages directory exists (fixture sanity)', () => {
    expect(existsSync(APP_ROOT)).toBe(true)
  })
})
