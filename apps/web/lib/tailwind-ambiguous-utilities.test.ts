/**
 * Repo invariant: `apps/web` source must not use the arbitrary-value form of
 * `ease-`, `duration-` or `delay-` — that is, one of those prefixes directly
 * followed by a bracketed value (shown in Tailwind's warning as `…[…]`).
 *
 * NB: this file must not spell those forms out either. `lib/**` is itself a
 * Tailwind content glob, so an example written here would be scanned and
 * re-emit the very warning this card silences — which is exactly what the
 * first draft of this canary did (8 build warnings, self-inflicted).
 *
 * Why these three prefixes and not others: `tailwindcss-animate@1.0.7`
 * re-registers `ease`, `duration` and `delay` for the *animation* properties
 * (animation-timing-function / animation-duration / animation-delay) while
 * Tailwind core registers the same three names for the *transition* properties.
 * Keyword classes (`ease-smooth`, `duration-300`) are unaffected — both
 * utilities emit and the cascade sorts it out — but an **arbitrary** value
 * matches both utilities, and Tailwind's ambiguity resolution then logs
 * "The class … is ambiguous and matches multiple utilities." and `continue`s:
 * **the rule is never emitted, with no build error.**
 *
 * Card a2bfd663: `ease-` combined with `cubic-bezier(0.16,1,0.3,1)` (ReasoningPanel
 * expand) and `ease-` combined with `cubic-bezier(0.222,0.133,0,1)` (AppLayout sidebar collapse)
 * produced no CSS at all — `dist-vite` shipped zero cubic-bezier rules — so
 * both animations silently ran on `.transition-all`'s default curve. The fix
 * is named keys under
 * `tailwind.config.cjs -> theme.extend.transitionTimingFunction` (`snappy`,
 * `glide`), which collide with nothing.
 *
 * This test is the canary: the next arbitrary value fails here instead of
 * shipping a dead class name. (The plugin's `duration`/`ease` registrations
 * themselves must stay — 30 `animate-in … duration-200` usages depend on the
 * animation-side copy, so stripping the collision is not an option.)
 */

import { readdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const webRoot = join(dirname(fileURLToPath(import.meta.url)), '..')

/** Build output, vendored specs and deps are not shipped UI. */
const NOT_SOURCE = new Set(['node_modules', 'dist', 'dist-vite', 'coverage', 'public', 'cypress'])

/** Same file types the config globs scan; `.cjs` configs are excluded on purpose. */
const SOURCE_RE = /\.(ts|tsx|js|jsx)$/

/** Ambiguous-with-tailwindcss-animate prefixes, arbitrary-value form only. */
const AMBIGUOUS_RE = /\b(?:ease|duration|delay)-\[/g

interface WalkState {
  hits: string[]
  scanned: number
}

function walk(dir: string, state: WalkState): void {
  let entries
  try {
    entries = readdirSync(dir, { withFileTypes: true })
  } catch {
    return
  }
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue
    const full = join(dir, entry.name)
    if (entry.isDirectory()) {
      if (NOT_SOURCE.has(entry.name)) continue
      walk(full, state)
    } else if (SOURCE_RE.test(entry.name)) {
      state.scanned += 1
      const matches = readFileSync(full, 'utf8').match(AMBIGUOUS_RE)
      if (matches) {
        const found = [...new Set(matches)].join(' ')
        state.hits.push(`${full.slice(webRoot.length + 1)}: ${found}`)
      }
    }
  }
}

describe('tailwind ambiguous utilities', () => {
  it('never uses ambiguous arbitrary-value ease/duration/delay classes in scanned source', () => {
    const state: WalkState = { hits: [], scanned: 0 }
    walk(webRoot, state)

    // Sanity: if the walk reads nothing the assertion below is vacuous.
    expect(state.scanned, 'source walk found no files to scan').toBeGreaterThan(50)
    expect(
      state.hits,
      `ambiguous arbitrary utilities (no CSS is emitted for these): ${state.hits.join(' | ')}`,
    ).toEqual([])
  })

  it('keeps strui source clean too - apps/web compiles it into the same CSS', () => {
    const state: WalkState = { hits: [], scanned: 0 }
    walk(join(webRoot, '../../packages/strui/src'), state)

    expect(state.scanned, 'strui walk found no files to scan').toBeGreaterThan(10)
    expect(state.hits, `ambiguous arbitrary utilities in strui: ${state.hits.join(' | ')}`).toEqual(
      [],
    )
  })
})
