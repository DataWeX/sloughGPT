import path from 'node:path'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
// Relative specifiers carry explicit .ts extensions: with `"type": "module"`
// the config graph is ESM, and the native config loader (Vite's future
// default) resolves ESM the way Node does — no extension search.
import { apiRoutesPlugin } from './vite/api-middleware-plugin.ts'
import { redirectsPlugin } from './vite/redirect-plugin.ts'

// Vite is the single web build stack (Next.js coexistence removed).
// Maps next/* onto local compat shims so App Router page components work unchanged
// (aliases remain as a safety net for codemoded client code).
//
// apiRoutesPlugin serves planner/calendar app/api route handlers in dev.
// redirectsPlugin 307s legacy paths (proxy.ts parity via lib/redirects.ts).
//
// Env: loadEnv reads .env.local / .env / .env.[mode] so NEXT_PUBLIC_API_URL
// points the client at the edge (:8080) without exporting shell vars.
const env = loadEnv('development', import.meta.dirname, '')
const apiUrl = env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const nodeEnv = env.NODE_ENV || process.env.NODE_ENV || 'development'

/**
 * Vite 8 applies `define` to bundled builds / SSR, but client modules in
 * classic dev are `isBundled: false` and the built-in define plugin skips
 * them. Inline `process.env.NEXT_PUBLIC_*` ourselves so .env.local wins
 * in the browser without `experimental.bundledDev`.
 */
function inlinePublicEnv(values: Record<string, string>) {
  return {
    name: 'inline-public-env',
    transform(code: string, id: string) {
      if (id.includes('node_modules')) return null
      let out = code
      let changed = false
      for (const [key, val] of Object.entries(values)) {
        const re = new RegExp(`process\\.env\\.${key}`, 'g')
        if (re.test(out)) {
          out = out.replace(re, JSON.stringify(val))
          changed = true
        }
      }
      return changed ? { code: out, map: null } : null
    },
  }
}

export default defineConfig({
  plugins: [
    react(),
    redirectsPlugin(),
    apiRoutesPlugin(),
    inlinePublicEnv({
      NEXT_PUBLIC_API_URL: apiUrl,
      NODE_ENV: nodeEnv,
    }),
  ],
  root: import.meta.dirname,
  publicDir: 'public',
  server: {
    host: true,
    port: 5173,
    strictPort: false,
  },
  resolve: {
    alias: [
      { find: /^@\//, replacement: path.resolve(import.meta.dirname) + '/' },
      {
        find: /^@sloughgpt\/strui$/,
        replacement: path.resolve(import.meta.dirname, '../../packages/strui/src'),
      },
      {
        find: /^next\/link$/,
        replacement: path.resolve(import.meta.dirname, 'vite/next-compat/link.tsx'),
      },
      {
        find: /^next\/navigation$/,
        replacement: path.resolve(import.meta.dirname, 'vite/next-compat/navigation.ts'),
      },
      {
        find: /^next\/dynamic$/,
        replacement: path.resolve(import.meta.dirname, 'vite/next-compat/dynamic.tsx'),
      },
      {
        find: /^next\/web-vitals$/,
        replacement: path.resolve(import.meta.dirname, 'vite/next-compat/web-vitals.ts'),
      },
      {
        find: /^next\/server$/,
        replacement: path.resolve(import.meta.dirname, 'vite/next-compat/server.ts'),
      },
      {
        find: /^next-auth\/react$/,
        replacement: path.resolve(import.meta.dirname, 'vite/next-compat/next-auth.tsx'),
      },
      {
        find: /^react$/,
        replacement: path.resolve(import.meta.dirname, '../../node_modules/react'),
      },
      {
        find: /^react-dom$/,
        replacement: path.resolve(import.meta.dirname, '../../node_modules/react-dom'),
      },
    ],
    dedupe: ['react', 'react-dom', 'scheduler'],
    preserveSymlinks: true,
  },
  define: {
    'process.env.NEXT_PUBLIC_API_URL': JSON.stringify(apiUrl),
    'process.env.NODE_ENV': JSON.stringify(nodeEnv),
  },
  build: {
    outDir: 'dist-vite',
    emptyOutDir: true,
  },
})
