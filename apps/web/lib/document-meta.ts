/** Document metadata shared by Vite index.html drift-guard (and formerly Next layout). */

export const metadata = {
  title: 'Slo - AI Platform',
  description: 'Enterprise-grade AI framework with production-ready ML infrastructure',
  icons: { icon: '/favicon.svg' },
} as const

/** Enables `env(safe-area-inset-*)` under notches / home indicators on mobile. */
export const viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
} as const
