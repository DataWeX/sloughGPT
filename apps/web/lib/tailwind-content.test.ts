/**
 * Repo invariant: every top-level source dir of `apps/web` must appear in
 * `tailwind.config.js` -> `content`.
 *
 * Tailwind only emits utilities it finds while scanning those globs. An omitted
 * dir does not fail the build — its classes are silently dropped from the
 * compiled CSS. Card bb8212a8: `features/` (the entire chat UI) was never
 * scanned, so 192 utilities went missing, `w-[var(--tool-panel-width)]` among
 * them, and the chat column measured 0px on screen while the Tools rail
 * overflowed to 1139px.
 *
 * This test is the canary: when a new top-level source dir appears, add its
 * glob here -> in the config instead of shipping half-styled UI.
 */

import { existsSync, readdirSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const webRoot = join(dirname(fileURLToPath(import.meta.url)), '..')
const nodeRequire = createRequire(import.meta.url)
const config = nodeRequire(join(webRoot, 'tailwind.config.js')) as { content?: string[] }

/** Build output and vendored specs are not shipped UI, so they are not scanned. */
const NOT_SOURCE = new Set(['node_modules', 'dist', 'dist-vite', 'coverage', 'public', 'cypress'])

const SOURCE_RE = /\.(ts|tsx|js|jsx)$/

function hasSource(dir: string): boolean {
  let entries
  try {
    entries = readdirSync(dir, { withFileTypes: true })
  } catch {
    return false
  }
  for (const entry of entries) {
    if (entry.isDirectory()) {
      if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue
      if (hasSource(join(dir, entry.name))) return true
    } else if (SOURCE_RE.test(entry.name)) {
      return true
    }
  }
  return false
}

describe('tailwind content globs', () => {
  const content = config.content ?? []

  it('exposes a non-empty content array', () => {
    expect(content.length).toBeGreaterThan(0)
  })

  it('scans every top-level source dir of apps/web', () => {
    const sourceDirs = readdirSync(webRoot, { withFileTypes: true })
      .filter(
        (entry) =>
          entry.isDirectory() && !entry.name.startsWith('.') && !NOT_SOURCE.has(entry.name),
      )
      .map((entry) => entry.name)
      .filter((name) => hasSource(join(webRoot, name)))

    // Sanity: if the walk finds nothing, the assertion below is vacuous.
    expect(sourceDirs.length).toBeGreaterThan(3)

    const uncovered = sourceDirs.filter(
      (name) => !content.some((glob) => glob.startsWith(`./${name}/`)),
    )

    expect(uncovered, `tailwind.config.js content misses: ${uncovered.join(', ')}`).toEqual([])
  })

  it('covers the chat feature that regressed (card bb8212a8)', () => {
    const canary = 'features/chat/components/panels/ChatToolPanel.tsx'
    expect(existsSync(join(webRoot, canary))).toBe(true)
    expect(content.some((glob) => glob.startsWith('./features/'))).toBe(true)
  })
})
