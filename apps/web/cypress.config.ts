import { defineConfig } from 'cypress'
import { loadEnv } from 'vite'

/**
 * E2E tests mock the FastAPI base URL (`NEXT_PUBLIC_API_URL`, default http://localhost:8000).
 * The value is read with the same loadEnv() call vite.config.ts uses and exposed
 * as `Cypress.env('apiUrl')`, so intercepts always target the origin the client
 * was built with (see cypress/support/api-base.ts).
 * Run: `npm run e2e:vite` or start `npm run dev` and `npm run e2e`.
 */
const envDir = typeof __dirname !== 'undefined' ? __dirname : process.cwd()
const env = loadEnv('development', envDir, '')
const apiUrl = env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default defineConfig({
  e2e: {
    baseUrl: 'http://localhost:5173',
    env: { apiUrl },
    video: false,
    screenshotOnRunFailure: true,
    defaultCommandTimeout: 15_000,
    pageLoadTimeout: 120000,
    experimentalMemoryManagement: true,
    numTestsKeptInMemory: 1,
    setupNodeEvents() {
      // extend plugins here if needed
    },
  },
})
