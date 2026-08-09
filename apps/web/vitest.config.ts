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
      ['contexts/**/*.test.{ts,tsx}', 'jsdom'],
      ['hooks/**/*.test.{ts,tsx}', 'jsdom'],
      ['app/**/*.test.{ts,tsx}', 'jsdom'],
      ['features/**/*.test.{ts,tsx}', 'jsdom'],
      ['lib/sync-html-theme.test.ts', 'jsdom'],
      ['lib/download-utils.test.ts', 'jsdom'],
      ['lib/reaction-store.test.ts', 'jsdom'],
      ['lib/query/hooks.test.ts', 'jsdom'],
      ['lib/query/api-hooks.test.ts', 'jsdom'],
      ['lib/soulnet-webgpu/**/*.test.ts', 'jsdom'],
    ],
    include: [
      'middleware.test.ts',
      'lib/**/*.test.ts',
      'contexts/**/*.test.{ts,tsx}',
      'hooks/**/*.test.{ts,tsx}',
      'components/**/*.test.{ts,tsx}',
      'app/**/*.test.{ts,tsx}',
      'features/**/*.test.{ts,tsx}',
    ],
    // ModelDetailPage.test.tsx excluded: vitest fork pool never exits because the
    // page's setInterval (uptime timer) keeps the Node event loop alive even after
    // cleanup() unmounts the component. Run separately:
    //   npx vitest run "app/(app)/model/[id]/ModelDetailPage.test.tsx"
    exclude: [
      'app/(app)/model/[id]/ModelDetailPage.test.tsx',
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
      '@sloughgpt/strui': path.resolve(__dirname, '../../packages/strui/src'),
    },
    dedupe: ['react', 'react-dom'],
  },
})
