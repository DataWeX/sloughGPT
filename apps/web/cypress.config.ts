import { defineConfig } from 'cypress'

/**
 * E2E tests mock the FastAPI base URL (`NEXT_PUBLIC_API_URL`, default http://localhost:8000).
 * Run: `npm run build && npm run e2e:ci` or start `npm run dev` and `npm run e2e`.
 */
export default defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000',
    video: false,
    screenshotOnRunFailure: true,
    defaultCommandTimeout: 15_000,
    setupNodeEvents() {
      // extend plugins here if needed
    },
  },
})
