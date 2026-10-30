import path from 'path'

import { defineConfig } from 'vitest/config'

const JSDOM_INCLUDE = [
  'components/**/*.test.{ts,tsx}',
  'contexts/**/*.test.{ts,tsx}',
  'hooks/**/*.test.{ts,tsx}',
  'app/**/*.test.{ts,tsx}',
  'features/**/*.test.{ts,tsx}',
  'lib/sync-html-theme.test.ts',
  'lib/download-utils.test.ts',
  'lib/reaction-store.test.ts',
  'lib/query/hooks.test.ts',
  'lib/query/api-hooks.test.ts',
  'lib/cache/hooks.test.ts',
  'lib/cache/api-hooks.test.ts',
  'lib/soulnet-webgpu/**/*.test.ts',
  'lib/protocol/useEndpoint.test.tsx',
  'vite/**/*.test.tsx',
]

const NODE_INCLUDE = ['lib/**/*.test.ts', 'lib/**/*.test.tsx', 'vite/**/*.test.ts']

const NODE_EXCLUDE = [
  'lib/sync-html-theme.test.ts',
  'lib/download-utils.test.ts',
  'lib/reaction-store.test.ts',
  'lib/query/hooks.test.ts',
  'lib/query/api-hooks.test.ts',
  'lib/cache/hooks.test.ts',
  'lib/cache/api-hooks.test.ts',
  'lib/soulnet-webgpu/**/*.test.ts',
  'lib/protocol/useEndpoint.test.tsx',
]

const JSDOM_EXCLUDE = ['app/(app)/model/[id]/ModelDetailPage.test.tsx']

export default defineConfig({
  esbuild: {
    jsx: 'automatic',
  },
  test: {
    projects: [
      {
        extends: true,
        test: {
          name: 'jsdom',
          environment: 'jsdom',
          include: JSDOM_INCLUDE,
          exclude: JSDOM_EXCLUDE,
          setupFiles: ['./vitest-setup.ts'],
        },
      },
      {
        extends: true,
        test: {
          name: 'node',
          environment: 'node',
          include: NODE_INCLUDE,
          exclude: NODE_EXCLUDE,
          setupFiles: ['./vitest-setup.ts'],
        },
      },
    ],
    passWithNoTests: true,
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
      react: path.resolve(__dirname, '../../node_modules/react'),
      'react/jsx-runtime': path.resolve(__dirname, '../../node_modules/react/jsx-runtime.js'),
      'react/jsx-dev-runtime': path.resolve(
        __dirname,
        '../../node_modules/react/jsx-dev-runtime.js',
      ),
      'react-dom': path.resolve(__dirname, '../../node_modules/react-dom'),
      'react-dom/client': path.resolve(__dirname, '../../node_modules/react-dom/client.js'),
      'react-dom/server': path.resolve(__dirname, '../../node_modules/react-dom/server.js'),
      'react-dom/test-utils': path.resolve(__dirname, '../../node_modules/react-dom/test-utils.js'),
      scheduler: path.resolve(__dirname, '../../node_modules/scheduler'),
    },
    dedupe: ['react', 'react-dom', 'scheduler'],
  },
})
