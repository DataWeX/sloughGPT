import path from 'path'

import { defineConfig } from 'vitest/config'

export default defineConfig({
  esbuild: {
    jsx: 'automatic',
  },
  test: {
    environment: 'node',
    include: ['lib/**/*.test.ts', 'hooks/**/*.test.{ts,tsx}', 'components/**/*.test.tsx', 'app/**/*.test.{ts,tsx}'],
    passWithNoTests: true,
    setupFiles: ['./vitest-setup.ts'],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '.'),
    },
  },
})
