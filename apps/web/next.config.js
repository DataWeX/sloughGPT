/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['@sloughgpt/strui'],
  // Disable Next.js default error overlay in development
  // We use our own CustomErrorHandler component instead
  onDemandEntries: {
    maxInactiveAge: 25 * 1000,
    pagesBufferLength: 2,
  },
  async rewrites() {
    return []
  },
}

module.exports = nextConfig
