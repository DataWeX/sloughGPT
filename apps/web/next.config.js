/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  transpilePackages: ['@sloughgpt/strui'],
  // Disable Next.js default error overlay in development
  // We use our own CustomErrorHandler component instead
  onDemandEntries: {
    maxInactiveAge: 25 * 1000,
    pagesBufferLength: 2,
  },
  async rewrites() {
    return [
      { source: '/auto-train/:path*', destination: 'http://localhost:8000/auto-train/:path*' }
    ]
  },
}

module.exports = nextConfig
