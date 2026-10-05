/**
 * Timeout budgets are an invariant, not a tuning detail.
 *
 * Card 21f45924: three full CI runs on a loaded box (load avg 20–36) flaked on
 * wall-clock budgets nobody had configured — vitest's beforeEach/afterEach was
 * still on the 10_000 default while `testTimeout` was 30_000 ("Hook timed out
 * in 10000ms"), and testing-library's `waitFor`/`findBy*` sat on their 1000ms
 * `asyncUtilTimeout`, the tightest budget in the suite: `VMPage` flaked with
 * 22ms of headroom left and `TokenTreeMergesCard` was caught mid-load. All four
 * affected specs passed in isolation with no code change, and the run at
 * load 13 came back 843/843 green — so the failures were the budgets, not the
 * tests.
 *
 * This test pins both budgets, so a future edit cannot silently restore the
 * defaults that produced those flakes.
 */
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { getConfig } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

const webRoot = join(dirname(fileURLToPath(import.meta.url)), '..')

/** Read the values straight out of the config source: vitest resolves the file
 *  through its own pipeline, so there is no runtime handle on the merged config. */
function declaredBudgets(): Record<string, number> {
  const config = readFileSync(join(webRoot, 'vitest.config.ts'), 'utf8')
  return Object.fromEntries(
    [...config.matchAll(/\b(testTimeout|hookTimeout):\s*([0-9_]+)/g)].map((match) => [
      match[1],
      Number(match[2].replace(/_/g, '')),
    ]),
  )
}

describe('test timeout budgets', () => {
  it('gives testing-library async utilities more than the historic 1s budget', () => {
    // Proves vitest-setup.ts actually ran configure(), not just that the number
    // is written down somewhere.
    expect(getConfig().asyncUtilTimeout).toBe(5_000)
  })

  it('keeps hookTimeout aligned with testTimeout', () => {
    const budgets = declaredBudgets()

    expect(budgets.testTimeout, 'testTimeout must stay declared').toBeGreaterThan(0)
    expect(budgets.hookTimeout, 'hookTimeout must stay declared').toBeGreaterThan(0)
    expect(
      budgets.hookTimeout,
      'hooks are the same class of slow work as tests — a hook budget below ' +
        'testTimeout reintroduces "Hook timed out" flakes under load',
    ).toBe(budgets.testTimeout)
  })
})
