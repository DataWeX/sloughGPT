import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { apiRoutesPlugin } from './vite/api-middleware-plugin'
import { redirectsPlugin } from './vite/redirect-plugin'

// Phase 0 migration spike — Vite coexists with Next.
// Maps next/* onto local compat shims so existing AppLayout/Sidebar work unchanged.
// Does NOT replace `next dev` / `next build`.
//
// Phase 2: apiRoutesPlugin serves planner/calendar app/api route handlers in dev.
// Phase 3: client code codemods onto @/vite/next-compat/* (aliases remain as safety net).
// Phase 4: redirectsPlugin 307s legacy paths (proxy.ts parity via lib/redirects.ts).

// Tree-identity header (card 20261029_041801): the dev server announces
// WHICH checkout it serves. Live-stack tests (tests/test_header_screenshots.py)
// compare this against their own repo root and SKIP — instead of asserting
// against a foreign vite squatting :5173 — whenever a different tree is
// answering. Mirrors web_tree_id() in tests/skip_gate_identity.py:
// sha256(realpath(apps/web)) truncated to 16 hex chars. Node's
// fs.realpathSync and Python's os.path.realpath both resolve symlinks, so
// the two sides agree byte-for-byte.
const webTreeId = crypto
  .createHash('sha256')
  .update(fs.realpathSync(__dirname))
  .digest('hex')
  .slice(0, 16)

export default defineConfig({
  plugins: [react(), redirectsPlugin(), apiRoutesPlugin()],
  root: __dirname,
  publicDir: 'public',
  server: {
    host: true,
    port: 5173,
    strictPort: false,
    headers: {
      'X-Slo-Web-Tree': webTreeId,
    },
    proxy: {
      // Single-origin dev: proxy FastAPI backend paths through Vite (5173).
      // Backend defaults to http://localhost:8000 (SLO_PORT).
      ...Object.fromEntries(
        [
          '/agents',
          '/auth',
          '/benchmark',
          '/chat',
          '/companion',
          '/config',
          '/consciousness',
          '/contracts',
          '/dashboard',
          '/datasets',
          '/docstore',
          '/docs',
          '/errors',
          '/experiments',
          '/feedback',
          '/files',
          '/health',
          '/images',
          '/infer',
          '/knowledge',
          '/learn',
          '/live',
          '/memory',
          '/meta-weights',
          '/mobile',
          '/model-stack',
          '/models',
          '/mole',
          '/multimodal',
          '/openapi.json',
          '/openwebui',
          '/plugins',
          '/profiles',
          '/rate-limit',
          '/ready',
          '/redoc',
          '/registry',
          '/search',
          '/security',
          '/session',
          '/settings',
          '/shell',
          '/souls',
          '/status',
          '/system',
          '/tenants',
          '/tokenizer',
          '/token-tree',
          '/tools',
          '/training',
          '/user-adapters',
          '/users',
          '/v1',
          '/vector',
          '/vm',
          '/voice',
          '/workflow',
          '/workspaces',
          '/world',
        ].map((p) => [
          p,
          {
            target: process.env.SLO_API_URL || 'http://localhost:8000',
            changeOrigin: true,
            ws: true,
          },
        ]),
      ),
    },
  },
  resolve: {
    alias: [
      { find: /^@\//, replacement: path.resolve(__dirname) + '/' },
      {
        find: /^@sloughgpt\/strui$/,
        replacement: path.resolve(__dirname, '../../packages/strui/src'),
      },
      { find: /^v86$/, replacement: path.resolve(__dirname, 'lib/node-builtin-stub.ts') },
      { find: /^react$/, replacement: path.resolve(__dirname, '../../node_modules/react') },
      { find: /^react-dom$/, replacement: path.resolve(__dirname, '../../node_modules/react-dom') },
    ],
    dedupe: ['react', 'react-dom', 'scheduler'],
    preserveSymlinks: true,
  },
  define: {
    'process.env.NEXT_PUBLIC_API_URL': JSON.stringify(
      // `||` would swallow an explicit empty string; an empty base makes the
      // static build same-origin (gateway relays the API — one origin, no CORS).
      process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000',
    ),
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
  },
  worker: {
    // The WebGPU worker is a module worker (lib/soulnet-webgpu/index.ts:62).
    // vite <=7 defaults worker.format to 'iife', which code-splitting builds
    // reject ("UMD and IIFE output formats are not supported"). Explicit 'es'
    // keeps the build green on both the root-resolved vite 7 and the
    // apps/web-pinned vite 8 (rolldown) toolchains.
    format: 'es',
  },
  build: {
    outDir: 'dist-vite',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        // Vendor split: keep the 806KB entry down to app code so first paint
        // doesn't parse react-dom/three/motion before any route runs. Vendor
        // chunks are immutable-cached; app entry changes on every release.
        manualChunks(id: string) {
          if (!id.includes('node_modules')) return undefined
          if (/node_modules\/(react|react-dom|scheduler)\//.test(id)) return 'vendor-react'
          if (/node_modules\/react-router/.test(id)) return 'vendor-router'
          if (/node_modules\/(framer-motion|motion)\//.test(id)) return 'vendor-motion'
          if (/node_modules\/(zustand|immer)\//.test(id)) return 'vendor-state'
          if (/node_modules\/(three|@react-three)\//.test(id)) return 'vendor-three'
          if (/node_modules\/recharts/.test(id)) return 'vendor-recharts'
          return 'vendor'
        },
      },
    },
  },
})
