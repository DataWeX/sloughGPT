// @vitest-environment node
/**
 * Server transport chokepoint gate (AGENTS.md — "http-client.ts is the
 * single source of truth for all API communication").
 *
 * The invariant: anything that communicates with the server passes through
 * ONE transport, so base URL, auth, correlation ids, retries, abort and the
 * error envelope are decided in exactly one place.
 *
 * The transport is NOT a single file — it is split across `apps/lib/`
 * (url, corr-id, interceptors, resilience, request, verbs, api-client,
 * client, sse, auth) and reached through the facade
 * `apps/web/lib/http-client.ts`.
 *
 * This test keeps two things honest:
 *
 *  - NO raw egress outside the facade. A file under apps/web calling
 *    `fetch(` / `new EventSource(` / `new WebSocket(` directly bypasses
 *    that policy and fails here unless it is in `knownRawEgress`.
 *  - The pin only SHRINKS: a pinned file that stops bypassing must have its
 *    entry removed (stale rows fail too), and `rawEgressCeiling` can never
 *    rise. Fixing a bypass = deleting a line, not editing a count.
 *
 * `apps/lib/` is deliberately NOT scanned: it IS the transport.
 */
import { describe, it, expect } from 'vitest'
import { readdirSync, readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import contract from './server-transport-contract.json'

const HERE = dirname(fileURLToPath(import.meta.url))
const WEB_ROOT = join(HERE, '..')
const APPS_LIB = join(WEB_ROOT, '..', 'lib')

/** Files that are allowed to talk to the server without the transport. */
const SCAN_ROOTS = ['app', 'components', 'hooks', 'contexts', 'features', 'lib', 'vite']

interface TransportContract {
  measuredOn: string
  rawEgressCeiling: number
  knownRawEgress: string[]
}

const pin = contract as TransportContract

/** Same expression the baseline was measured with — comments and strings
 *  are stripped first so prose like "auto-fetch (manual refetch only)" and
 *  type signatures never count as egress. */
const EGRESS =
  /(?<![\w$.:=-])(?:fetch|window\.fetch)\s*\(|new\s+EventSource\s*\(|new\s+WebSocket\s*\(/

function stripComments(src: string): string {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/.*$/gm, '')
}

function scan(): string[] {
  const out: string[] = []
  const walk = (rel: string): void => {
    const abs = join(WEB_ROOT, rel)
    let entries
    try {
      entries = readdirSync(abs, { withFileTypes: true })
    } catch {
      return
    }
    for (const e of entries) {
      const childRel = `${rel}/${e.name}`
      if (e.isDirectory()) {
        if (e.name !== 'node_modules') walk(childRel)
        continue
      }
      if (!/\.(ts|tsx)$/.test(e.name) || /\.test\.(ts|tsx)$/.test(e.name)) continue
      const src = stripComments(readFileSync(join(WEB_ROOT, childRel), 'utf8'))
      EGRESS.lastIndex = 0
      if (EGRESS.test(src)) out.push(childRel)
    }
  }
  for (const root of SCAN_ROOTS) walk(root)
  return out.sort()
}

const onDisk = scan()
const pinned = [...pin.knownRawEgress].sort()

describe('Server transport — raw egress chokepoint', () => {
  it('has no raw egress file outside the pin', () => {
    const unlisted = onDisk.filter((p) => !pin.knownRawEgress.includes(p))
    expect(
      unlisted,
      `These talk to the server without the transport — route them through ` +
        `lib/http-client.ts (apiGet/apiPost/streamSSE/authFetch) instead of ` +
        `adding them to the pin:\n  ${unlisted.join('\n  ')}`
    ).toEqual([])
  })

  it('pins no stale rows (fixed files must leave the pin)', () => {
    const stale = pin.knownRawEgress.filter((p) => !onDisk.includes(p))
    expect(
      stale,
      `These no longer do raw egress — delete their entries so the pin shrinks:\n  ${stale.join('\n  ')}`
    ).toEqual([])
  })

  it('is exactly the disk state (no drift in either direction)', () => {
    expect(pinned).toEqual(onDisk)
  })

  it('never exceeds the measured ceiling', () => {
    expect(onDisk.length).toBeLessThanOrEqual(pin.rawEgressCeiling)
    expect(pin.knownRawEgress.length).toBeLessThanOrEqual(pin.rawEgressCeiling)
  })

  it('keeps the facade wired to the split transport', () => {
    const facade = readFileSync(join(HERE, 'http-client.ts'), 'utf8')
    expect(facade).toContain("export * from '../../lib/index'")

    const barrel = readFileSync(join(APPS_LIB, 'index.ts'), 'utf8')
    // REST verbs, streaming and auth must all stay reachable through it.
    for (const surface of [
      'apiGet',
      'apiPost',
      'apiPut',
      'apiDelete',
      'apiPatch',
      'streamSSE',
      'authFetch',
    ]) {
      expect(barrel, `transport no longer exposes ${surface}`).toContain(surface)
    }
  })

  it('never imports the app upward — ports, not upward imports', () => {
    const UPWARD = /from ['"]\.\.\/web\/lib/
    const leaks: string[] = []
    for (const entry of readdirSync(APPS_LIB, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.endsWith('.ts') || entry.name.includes('.test.'))
        continue
      if (UPWARD.test(readFileSync(join(APPS_LIB, entry.name), 'utf8'))) {
        leaks.push(`apps/lib/${entry.name}`)
      }
    }
    expect(
      leaks,
      `apps/lib must never import apps/web: it couples the transport to one host ` +
        `app and blocks reuse by any other. Supply the value through ` +
        `apps/lib/ports.ts (wired by configureTransport in ` +
        `apps/web/lib/http-client.ts) instead:\n  ${leaks.join('\n  ')}`
    ).toEqual([])
  })
})
