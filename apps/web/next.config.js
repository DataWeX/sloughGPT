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
  eslint: { ignoreDuringBuilds: false },
  distDir: process.env.BUILD_DIST || (process.env.NODE_ENV === 'development' ? '.next-dev' : '.next'),
  transpilePackages: ['@sloughgpt/strui'],
  images: {
    formats: ['image/avif', 'image/webp'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
  },
  experimental: {
    optimizeCss: true,
    modularizeImports: {
      'lucide-react': {
        transform: 'lucide-react/dist/esm/icons/{{ kebabCase member }}',
      },
      'recharts': {
        transform: 'recharts/{{ member }}',
      },
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
