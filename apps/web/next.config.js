/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  ...(process.env.NODE_ENV === 'production' && { output: 'standalone' }),
  onDemandEntries: {
    maxInactiveAge: 25 * 1000,
    pagesBufferLength: 2,
  },
  typescript: { ignoreBuildErrors: true },
  eslint: { ignoreDuringBuilds: true },
  distDir: process.env.BUILD_DIST || (process.env.NODE_ENV === 'development' ? '.next-dev' : '.next'),
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
