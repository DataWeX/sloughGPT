import { defineConfig } from 'cypress'
import fs from 'fs'

export default defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3010',
    video: false,
    screenshotOnRunFailure: true,
    defaultCommandTimeout: 15_000,
    pageLoadTimeout: 120000,
    experimentalMemoryManagement: true,
    numTestsKeptInMemory: 1,
    setupNodeEvents(on) {
      on('task', {
        writeHydrationDump(html: string) {
          fs.writeFileSync('/tmp/home_client.json', html)
          return null
        },
        writeHydrationDumpBefore(html: string) {
          fs.writeFileSync('/tmp/home_before.json', html)
          return null
        },
      })
    },
  },
})
