/**
 * Repo invariant: UI code must never call `new Date(x).toLocale*()` directly.
 *
 * Raw calls render "Invalid Date" whenever the backend sends an unparsable
 * timestamp (the legacy `…+00:00Z` format caused the souls-page bug fixed in
 * card 054). All date rendering goes through a total formatter —
 * `@/lib/time-format`, `apps/mobile` `format-utils`, or strui `format-time` —
 * which returns `''` for input it cannot parse.
 *
 * `new Date()` (no argument, i.e. "now") is always valid and allowed.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const HERE = dirname(fileURLToPath(import.meta.url))

/** Walk up until we find the repo root (has AGENTS.md + apps/). */
function repoRoot(from: string): string {
  let dir = from
  for (let i = 0; i < 8; i++) {
    try {
      readFileSync(join(dir, 'AGENTS.md'), 'utf-8')
      if (statSync(join(dir, 'apps')).isDirectory()) return dir
    } catch {
      /* keep walking */
    }
    dir = dirname(dir)
  }
  throw new Error('repo root not found from ' + from)
}

const ROOT = repoRoot(HERE)

const SCAN_DIRS = ['apps/web', 'apps/mobile/src', 'packages/strui/src']

/** The formatter libraries themselves are allowed to call `new Date`. */
const ALLOWED = new Set([
  'apps/web/lib/time-format.ts',
  'apps/web/lib/formatDuration.ts',
  'apps/web/lib/format-bytes.ts',
  'apps/web/lib/time-ago.ts',
  'apps/mobile/src/services/format-utils.ts',
  'packages/strui/src/lib/format-time.ts',
])

const EXT = new Set(['.ts', '.tsx', '.js', '.jsx', '.mjs'])

const SKIP_DIRS = new Set([
  'node_modules',
  '__pycache__',
  'dist-vite',
  'dist',
  'build',
  'coverage',
  '.next',
  '.turbo',
])

function walk(dir: string, out: string[] = []): string[] {
  let entries: string[]
  try {
    entries = readdirSync(dir)
  } catch {
    return out
  }
  for (const name of entries) {
    if (name.startsWith('.') || SKIP_DIRS.has(name)) continue
    const full = join(dir, name)
    const st = statSync(full)
    if (st.isDirectory()) walk(full, out)
    else if (EXT.has(name.slice(name.lastIndexOf('.')))) out.push(full)
  }
  return out
}

/** Return `${relPath}:${line}` for each raw `new Date(<non-empty>).to[Ll]ocale…` call. */
function scan(file: string, text: string): string[] {
  const hits: string[] = []
  const needle = 'new Date('
  let i = text.indexOf(needle)
  while (i >= 0) {
    const open = i + needle.length - 1 // index of '('
    let depth = 0
    let j = open
    let quote: string | null = null
    while (j < text.length) {
      const c = text[j]
      if (quote) {
        if (c === '\\') j += 2
        else if (c === quote) quote = null
        j += 1
        continue
      }
      if (c === "'" || c === '"' || c === '`') quote = c
      else if (c === '(') depth += 1
      else if (c === ')') {
        depth -= 1
        if (depth === 0) break
      }
      j += 1
    }
    const close = j
    const arg = text.slice(open + 1, close).trim()
    const rest = text.slice(close + 1)
    if (arg !== '' && /^\s*\.\s*to[Ll]ocale/.test(rest)) {
      hits.push(`${file.replace(ROOT + '/', '')}:${text.slice(0, i).split('\n').length}`)
    }
    i = text.indexOf(needle, i + needle.length)
  }
  return hits
}

describe('no raw new Date(x).toLocale* in UI code', () => {
  it('routes every date render through a total formatter', () => {
    const offenders: string[] = []
    for (const dir of SCAN_DIRS) {
      for (const file of walk(join(ROOT, dir))) {
        const rel = file.replace(ROOT + '/', '')
        if (ALLOWED.has(rel)) continue
        if (rel.includes('.test.')) continue
        let text: string
        try {
          text = readFileSync(file, 'utf-8')
        } catch {
          continue
        }
        if (!text.includes('new Date(') || !text.includes('toLocale')) continue
        if (rel.includes('format-utils') || rel.includes('format-time')) continue
        offenders.push(...scan(rel, text))
      }
    }
    expect(
      offenders,
      'Use the guarded formatters (@/lib/time-format, mobile format-utils, strui format-time) instead of new Date(...).toLocale* — raw calls render "Invalid Date" for unparsable backend timestamps.',
    ).toEqual([])
  })
})
