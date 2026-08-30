/** @type {import('next').NextConfig} */
const isProd = process.env.NODE_ENV === 'production'
const isExport = process.env.NEXT_EXPORT === '1'

const nextConfig = {
  reactStrictMode: true,
  ...(isProd && !isExport && { output: 'standalone' }),
  ...(isExport && { output: 'export', images: { unoptimized: true }, trailingSlash: true }),
  onDemandEntries: {
    maxInactiveAge: 25 * 1000,
    pagesBufferLength: 2,
  },
  typescript: { ignoreBuildErrors: true },
  distDir: process.env.BUILD_DIST || (process.env.NODE_ENV === 'development' ? '.next-dev' : '.next'),
  transpilePackages: ['@sloughgpt/strui'],
}

module.exports = nextConfig
