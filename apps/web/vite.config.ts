import path from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { redirectsPlugin } from './vite/redirect-plugin'

// Phase 0 migration spike — Vite coexists with Next.
// Maps next/* onto local compat shims so existing AppLayout/Sidebar work unchanged.
// Does NOT replace `next dev` / `next build`.
//
// Phase 2 (updated): the planner/calendar app/api route handlers were ported to
// the Python backend (apps/api/server/routers/planner.py + calendar.py), so dev
// proxies /api to FastAPI exactly like the deployed gateway relays it. The old
// apiRoutesPlugin (in-process Next handler server) is gone.
// Phase 3: client code codemods onto @/vite/next-compat/* (aliases remain as safety net).
// Phase 4: redirectsPlugin 307s legacy paths (proxy.ts parity via lib/redirects.ts).

// Dev API target: same default as the NEXT_PUBLIC_API_URL define below.
// `||` (not `??`) here — an empty target would proxy /api back to this server
// (infinite loop); the build-time define keeps its own `??` semantics.
const apiTarget = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default defineConfig({
  plugins: [react(), redirectsPlugin()],
  root: __dirname,
  publicDir: 'public',
  server: {
    host: true,
    port: 5173,
    strictPort: false,
    // Same-origin /api/* in dev (apps/lib/url.ts keeps planner/calendar
    // same-origin) — forward to the backend instead of the old apiRoutesPlugin.
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: [
      { find: /^@\//, replacement: path.resolve(__dirname) + '/' },
      {
        find: /^@sloughgpt\/strui$/,
        replacement: path.resolve(__dirname, '../../packages/strui/src'),
      },
      { find: /^next\/link$/, replacement: path.resolve(__dirname, 'vite/next-compat/link.tsx') },
      {
        find: /^next\/navigation$/,
        replacement: path.resolve(__dirname, 'vite/next-compat/navigation.ts'),
      },
      {
        find: /^next\/dynamic$/,
        replacement: path.resolve(__dirname, 'vite/next-compat/dynamic.tsx'),
      },
      {
        find: /^next\/web-vitals$/,
        replacement: path.resolve(__dirname, 'vite/next-compat/web-vitals.ts'),
      },
      {
        find: /^next\/server$/,
        replacement: path.resolve(__dirname, 'vite/next-compat/server.ts'),
      },
      {
        find: /^next-auth\/react$/,
        replacement: path.resolve(__dirname, 'vite/next-compat/next-auth.tsx'),
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
  build: {
    outDir: 'dist-vite',
    emptyOutDir: true,
  },
  ssr: {
    external: ['next'],
  },
})
