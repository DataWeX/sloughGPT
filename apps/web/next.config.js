/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  transpilePackages: ['@sloughgpt/strui'],
  async rewrites() {
    return [
      { source: '/auto-train/:path*', destination: 'http://localhost:8000/auto-train/:path*' }
    ]
  },
}

module.exports = nextConfig
