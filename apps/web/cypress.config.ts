import { writeFileSync } from 'node:fs'
import { join } from 'node:path'

import { defineConfig } from 'cypress'

/**
 * E2E tests mock the API same-origin (card 20261123_135845): the client sends
 * root-relative requests, so intercepts default to a '' base
 * (cypress/support/api-base.ts). Point at a real backend with
 * `cypress run --env apiUrl=http://127.0.0.1:8001` when needed.
 * Run: `npm run e2e:vite` or start `npm run dev` and `npm run e2e`.
 */

export default defineConfig({
  e2e: {
    baseUrl: 'http://localhost:5173',
    video: false,
    screenshotOnRunFailure: true,
    defaultCommandTimeout: 15_000,
    pageLoadTimeout: 120000,
    experimentalMemoryManagement: true,
    numTestsKeptInMemory: 1,
    setupNodeEvents(on) {
      // zz-hydration-diag.cy.ts dumps post-hydration head/body for debugging.
      on('task', {
        writeHydrationDump(contents: string) {
          writeFileSync(join(envDir, '.hydration-dump.json'), contents)
          return null
        },
      })
    },
  },
})
