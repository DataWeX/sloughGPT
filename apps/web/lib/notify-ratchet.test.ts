import { describe, it, expect } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'

/**
 * notify() ratchet (card 20261107_064201, reconciled 2026-11-14) — ZERO baseline.
 *
 * Every notification WRITE in apps/web must go through @/lib/notify — the
 * `lib/notify/` DIRECTORY facade (main's former `lib/notify.ts` dispatcher
 * was removed in the reconciliation: module resolution prefers `notify.ts`
 * over `notify/index.ts`, so a re-added file would silently shadow the whole
 * library — the sanity check below pins that shut).
 *
 * This gate scans every non-test source file for the ways code can bypass
 * the handle:
 *   1. acquiring a tier store's write method (selector / getState line),
 *   2. calling a write method off a store/state reference directly,
 *   3. importing a deleted pre-reconciliation store (toast-store/banner-store),
 *   4. positional toast calls (the old addToast(msg, type) shape — the
 *      facade takes an options object; tsc is the airtight backstop, this
 *      rule is the readable failure message),
 * plus any dock projector intake (`ingestProcessEvents`) outside the dock's
 * own module.
 *
 * Reads are not violations: the tier views subscribe to state (GlobalBanner
 * reads `banners[0]`, AppLayout reads `toasts`) — only the write half is
 * sealed. Same discipline as the page-baseline / handbuilt-routes /
 * no-raw-threads gates: policy lives in docs, this keeps sessions honest
 * at 2am.
 *
 * EXEMPT: `lib/notify/**` (the handle itself + the tiers it delegates to),
 * `lib/process-notifications.ts` (the dock projector defines its own
 * intake), test files (mocking the sink IS the test technique).
 */

const WEB_ROOT = path.resolve(__dirname, '..')

const isExempt = (rel: string): boolean =>
  rel.startsWith('lib/notify/') || rel === 'lib/process-notifications.ts'

const isTestFile = (rel: string) =>
  /\.test\.|\.spec\.|__tests__\//.test(rel) || rel.startsWith('cypress/')

/** Bypass patterns — kept as (label, RegExp) pairs for honest failure output. */
const RULES: Array<[string, RegExp]> = [
  ['direct toast write off a store/state ref', /\.addToast\(/],
  [
    'toast store write-method acquisition',
    /useToastStore[^\n]*(addToast|dismissToast|clearToasts)/,
  ],
  ['direct toast dismiss/clear off a store ref', /(?<!notify)\.(dismissToast|clearToasts)\(/],
  [
    'banner store write-method acquisition',
    /useBannerStore[^\n]*(showBanner|dismissBanner|clearBanners)/,
  ],
  ['direct banner write off a store ref', /(?<!notify)\.(showBanner|dismissBanner|clearBanners)\(/],
  ['import of a deleted pre-reconciliation store', /from ['"][^'"]*(toast-store|banner-store)['"]/],
  // positional = a quoted string as FIRST arg followed by a quoted second
  // arg (branch A: simple first args; branch B: first-arg strings containing
  // commas). Options-style calls (`{ type: … }`, nested calls) match neither.
  [
    'positional toast call (options object required)',
    /notify\.toast\((?:[^(),]*,\s*['"`]|(['"`])(?:[^'"`\\]|\\.)*\1\s*,\s*['"`])/,
  ],
  ['dock projector intake outside the dock', /ingestProcessEvents/],
]

function collectViolations(): string[] {
  const violations: string[] = []
  const walk = (dir: string): void => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue
      const abs = path.join(dir, entry.name)
      if (entry.isDirectory()) {
        walk(abs)
        continue
      }
      if (!/\.(ts|tsx)$/.test(entry.name)) continue
      const rel = path.relative(WEB_ROOT, abs)
      if (isExempt(rel) || isTestFile(rel)) continue
      const lines = fs.readFileSync(abs, 'utf8').split('\n')
      lines.forEach((line, i) => {
        for (const [label, re] of RULES) {
          if (re.test(line)) {
            violations.push(`${rel}:${i + 1} [${label}] ${line.trim().slice(0, 110)}`)
            break
          }
        }
      })
    }
  }
  walk(WEB_ROOT)
  return violations
}

describe('notify ratchet — zero baseline (no store-direct notification writes)', () => {
  it('every notification write in apps/web goes through @/lib/notify', () => {
    const violations = collectViolations()
    expect(
      violations,
      `direct tier-store writes bypass notify() — route them through @/lib/notify:\n${violations.join('\n')}`,
    ).toEqual([])
  })

  it('the directory facade is present and the shadow dispatcher stays gone', () => {
    expect(fs.existsSync(path.join(WEB_ROOT, 'lib/notify/index.ts'))).toBe(true)
    // module resolution prefers lib/notify.ts over lib/notify/ — a re-added
    // dispatcher would silently shadow the whole library (2026-11-14 trap).
    expect(fs.existsSync(path.join(WEB_ROOT, 'lib/notify.ts'))).toBe(false)
    expect(isExempt('lib/notify/index.ts')).toBe(true)
  })
})
