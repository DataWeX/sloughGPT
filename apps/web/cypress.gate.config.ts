import { defineConfig } from 'cypress'

import baseConfig from './cypress.config'

/**
 * Gate config for the expected-failure controls (card 20261125_130530).
 *
 * Why this file exists: `excludeSpecPattern` filters BOTH the default run and
 * an explicit `--spec` (verified: pointing `--spec` at an excluded spec makes
 * Cypress report "Can't run because no spec files were found" and exit 1 —
 * which an inverted gate would read as "failures surfaced", a false pass).
 * So the gate cannot be a `--spec` override of the main config; it needs a
 * config where the controls are the *only* specs and nothing excludes them.
 *
 * Everything else is inherited from cypress.config.ts so the controls run
 * under exactly the same support file, timeouts and tasks as the real suite —
 * that is the whole point: the thing under test is the global handler set in
 * cypress/support/e2e.ts.
 */
const baseE2E = baseConfig.e2e ?? {}

export default defineConfig({
  ...baseConfig,
  e2e: {
    ...baseE2E,
    specPattern: 'cypress/e2e/gate/**/*.cy.ts',
    excludeSpecPattern: [],
    // The controls are supposed to fail — a screenshot per failure is noise.
    screenshotOnRunFailure: false,
  },
})
