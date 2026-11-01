/** @type {import('next').NextConfig} */
const path = require('path')
const isProd = process.env.NODE_ENV === 'production'
const isExport = process.env.NEXT_EXPORT === '1'

const nextConfig = {
  outputFileTracingRoot: path.join(__dirname, '../../'),
  reactStrictMode: true,
  productionBrowserSourceMaps: process.env.SOURCE_MAPS === '1',
  ...(isProd && !isExport && { output: 'standalone' }),
  ...(isExport && { output: 'export', images: { unoptimized: true }, trailingSlash: true }),
  onDemandEntries: {
    maxInactiveAge: 25 * 1000,
    pagesBufferLength: 2,
  },
  typescript: { ignoreBuildErrors: false },
  distDir: process.env.BUILD_DIST || (process.env.NODE_ENV === 'development' ? '.next-dev' : '.next'),
  transpilePackages: ['@sloughgpt/strui'],
  images: {
    formats: ['image/avif', 'image/webp'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
  },
  modularizeImports: {
    'lucide-react': {
      transform: 'lucide-react/dist/esm/icons/{{ kebabCase member }}',
    },
    // NOTE: recharts must NOT be modularized — v2 has no per-component
    // subpath exports (recharts/Area etc. do not resolve).
  },
  experimental: {},
  turbopack: {
    // `next dev` runs on Turbopack where the `webpack` key below is ignored.
    // The v86 ESM build pulls node builtins (fs, perf_hooks) that cannot
    // resolve in the browser bundle. Alias the package to a stub for browser
    // environments only (server/middleware bundles are untouched) — mirrors
    // the webpack `externals: { v86: 'v86' }` handling used by `next build`.
    // The stub only surfaces if the in-browser VM is actually initialized.
    resolveAlias: {
      v86: { browser: './lib/node-builtin-stub.ts' },
    },
  },
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      }
    }
    // v86 ESM build imports node: builtins — externalize to avoid webpack errors
    config.externals = [
      ...(Array.isArray(config.externals) ? config.externals : []),
      { 'v86': 'v86' },
    ]
    return config
  },
}

module.exports = nextConfig
