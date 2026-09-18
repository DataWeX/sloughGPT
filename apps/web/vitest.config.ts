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
      ['lib/cache/hooks.test.ts', 'jsdom'],
      ['lib/cache/api-hooks.test.ts', 'jsdom'],
      ['lib/soulnet-webgpu/**/*.test.ts', 'jsdom'],
    ],
    include: [
      'proxy.test.ts',
      'lib/**/*.test.ts',
      'contexts/**/*.test.{ts,tsx}',
      'hooks/**/*.test.{ts,tsx}',
      'components/**/*.test.{ts,tsx}',
      'app/**/*.test.{ts,tsx}',
      'features/**/*.test.{ts,tsx}',
    ],
    passWithNoTests: true,
    exclude: ['app/(app)/model/[id]/ModelDetailPage.test.tsx'],
    setupFiles: ['./vitest-setup.ts'],
    typecheck: {
      tsconfig: './vitest.tsconfig.json',
    },
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
      react: path.resolve(__dirname, 'node_modules/react'),
      'react/jsx-runtime': path.resolve(__dirname, 'node_modules/react/jsx-runtime.js'),
      'react/jsx-dev-runtime': path.resolve(__dirname, 'node_modules/react/jsx-dev-runtime.js'),
      'react-dom': path.resolve(__dirname, 'node_modules/react-dom'),
      'react-dom/client': path.resolve(__dirname, 'node_modules/react-dom/client.js'),
      'react-dom/server': path.resolve(__dirname, 'node_modules/react-dom/server.js'),
      'react-dom/test-utils': path.resolve(__dirname, 'node_modules/react-dom/test-utils.js'),
      scheduler: path.resolve(__dirname, 'node_modules/scheduler'),
    },
    dedupe: ['react', 'react-dom', 'scheduler'],
  },
})
