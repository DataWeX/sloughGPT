import { describe, it, expect } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'

/**
 * Frontend raw-fetch gate (card 20261108_142919) — ZERO baseline for new code.
 *
 * Every API call in apps/web must go through the http-client facade
 * (`apiGet/apiPost/...` → `apps/lib/request.ts`), which owns retries,
 * circuit breaking, correlation IDs, auth headers and error envelopes.
 * A raw `fetch()` bypasses all of that — and, more importantly, is the
 * class of call the card's threat model ("where do we fetch TO") wants
 * forced through one auditable seam.
 *
 * The gate scans every non-test .ts/.tsx file under apps/web for raw
 * fetch spellings, with comments and `.fetch()` method calls (store
 * selectors like `operationsStore.getState().fetch()`) excluded.
 *
 * EXEMPT — grandfathered files with a reason (shrink-only; the honesty
 * test fails when an entry stops having a raw fetch, so entries must be
 * removed as call sites migrate):
 */
const ALLOWED: Record<string, string> = {
  'hooks/useV86.ts': 'v86 machine assets — binary payloads, no API envelope',
  'lib/soulnet-webgpu/engine.ts': 'webgpu model artifacts fetched as binary blobs',
  'lib/soulnet-webgpu/transformer-engine.ts': 'webgpu model artifacts fetched as binary blobs',
  'lib/soulnet-webgpu/worker.ts': 'webgpu model artifacts fetched as binary blobs',
  'lib/multimodal-controller.ts': 'data: URL image fetch — no network destination',
  'app/(app)/consciousness/api-explorer/page.tsx':
    'api-explorer interactive console (grandfathered surface)',
  'app/(app)/consciousness/docs/docs-endpoint-list.tsx':
    'fetch( appears inside a documentation code-sample string',
  'features/chat/hooks/useChatLocalEngine.ts': 'local-model asset fetch (no API envelope)',
  'features/chat/components/panels/useVisionStudio.ts':
    'data:/blob vision asset fetch — no network destination',
  'lib/planner-proxy.ts':
    'server-side planner relay — the one sanctioned raw fetch (Next route runtime)',
}

const WEB_ROOT = path.resolve(__dirname, '..')

const isTestFile = (rel: string) =>
  /\.test\.|\.spec\.|__tests__\//.test(rel) || rel.startsWith('cypress/')

/** Bypass spellings — (label, RegExp) pairs for honest failure output. */
const RULES: Array<[string, RegExp]> = [
  ['raw fetch() bypassing the http-client facade', /(?<![\w.$])fetch\s*\(/],
  ['raw fetch via window/globalThis/self', /\b(?:globalThis|window|self)\s*\.\s*fetch\s*\(/],
]

/** Comments carry example fetches in docs/annotations — not call sites. */
const isCommentLine = (line: string): boolean => /^\s*(\/\/|\*|\/\*)/.test(line)

function scanLines(rel: string, lines: string[]): string[] {
  const violations: string[] = []
  lines.forEach((line, i) => {
    if (isCommentLine(line)) return
    for (const [label, re] of RULES) {
      if (re.test(line)) {
        violations.push(`${rel}:${i + 1} [${label}] ${line.trim().slice(0, 110)}`)
        break
      }
    }
  })
  return violations
}

/** The full gate decision for one file: allowlist first, then line scan. */
function violationsFor(rel: string, content: string): string[] {
  if (ALLOWED[rel]) return []
  return scanLines(rel, content.split('\n'))
}

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
      if (isTestFile(rel)) continue
      violations.push(...violationsFor(rel, fs.readFileSync(abs, 'utf8')))
    }
  }
  walk(WEB_ROOT)
  return violations
}

describe('fetch gate — raw fetch outside the http-client facade (card 20261108_142919)', () => {
  it('zero baseline: every raw fetch is allowlisted with a reason', () => {
    const violations = collectViolations()
    expect(
      violations,
      `raw fetch() bypasses the http-client facade — migrate to apiGet/apiPost ` +
        `or justify an allowlist entry:\n${violations.join('\n')}`,
    ).toEqual([])
  })

  it('catches a synthetic violation — the gate can fail', () => {
    const v = violationsFor(
      'features/new-thing.ts',
      'export async function push() {\n  const r = await fetch("/api/x", { method: "POST" })\n  return r\n}\n',
    )
    expect(v).toHaveLength(1)
    expect(v[0]).toContain('features/new-thing.ts:2')
    expect(v[0]).toContain('bypassing the http-client facade')
  })

  it('catches the window/globalThis spellings', () => {
    expect(violationsFor('a.ts', 'window.fetch("/x")')).toHaveLength(1)
    expect(violationsFor('b.ts', 'globalThis.fetch("/x")')).toHaveLength(1)
  })

  it('ignores comments, .fetch() method calls, and the facade itself', () => {
    expect(
      violationsFor(
        'c.ts',
        [
          '// we deliberately do not call fetch() here',
          ' * example: fetch("/x") in a doc comment',
          'operationsStore.getState().fetch()',
          'await apiPost("/x", { a: 1 })',
          'const prefix = "myfetch";',
        ].join('\n'),
      ),
    ).toEqual([])
  })

  it('allowlisted files skip the scan (the exemption is real)', () => {
    expect(violationsFor('hooks/useV86.ts', 'await fetch(url)')).toEqual([])
  })

  it('the allowlist stays honest: entries exist and still carry a raw fetch', () => {
    for (const [rel, reason] of Object.entries(ALLOWED)) {
      expect(reason.length, `allowlist entry missing a reason: ${rel}`).toBeGreaterThan(10)
      const abs = path.join(WEB_ROOT, rel)
      expect(fs.existsSync(abs), `stale allowlist entry (file gone): ${rel}`).toBe(true)
      const lines = fs.readFileSync(abs, 'utf8').split('\n')
      expect(
        scanLines(rel, lines),
        `allowlist entry no longer has a raw fetch — remove it: ${rel}`,
      ).not.toHaveLength(0)
    }
  })
})
