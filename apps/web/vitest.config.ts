import path from 'path'

import { defineConfig } from 'vitest/config'

export default defineConfig({
  esbuild: {
    jsx: 'automatic',
  },
  test: {
    environment: 'node',
    environmentMatchGlobs: [
      ['components/**/*.test.{ts,tsx}', 'jsdom'],
      ['hooks/**/*.test.{ts,tsx}', 'jsdom'],
      ['app/**/*.test.{ts,tsx}', 'jsdom'],
    ],
    include: [
      'lib/**/*.test.ts',
      'hooks/**/*.test.{ts,tsx}',
      'components/**/*.test.{ts,tsx}',
      'app/**/*.test.{ts,tsx}',
    ],
    passWithNoTests: true,
    setupFiles: ['./vitest-setup.ts'],
    pool: 'forks',
    poolOptions: {
      forks: {
        singleFork: false,
        minForks: 1,
        maxForks: 4,
      },
    },
    clearMocks: true,
    testTimeout: 30_000,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '.'),
    },
  },
})
