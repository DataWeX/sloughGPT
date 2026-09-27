/** Document metadata shared by Vite index.html drift-guard (and formerly Next layout). */

export const metadata = {
  title: 'Man - AI Platform',
  description:
    'Man — your personal AI. Train it on your own data, then chat, write and learn together.',
  icons: { icon: '/favicon.svg' },
  manifest: '/manifest.json',
} as const

/** Enables `env(safe-area-inset-*)` under notches / home indicators on mobile. */
export const viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
} as const
