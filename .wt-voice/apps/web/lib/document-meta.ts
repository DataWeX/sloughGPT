/** Document metadata shared by Vite index.html drift-guard (and formerly Next layout). */

export const metadata = {
  title: 'Man - AI Platform',
  description:
    'Man — your personal AI. Train it on your own data, then chat, write and learn together.',
  icons: { icon: '/favicon.svg' },
  manifest: '/manifest.json',
  openGraph: {
    title: 'Man - AI Platform',
    description:
      'Man — your personal AI. Train it on your own data, then chat, write and learn together.',
    type: 'website',
    siteName: 'Man - AI Platform',
    locale: 'en_US',
    image: '/og.png',
    imageWidth: '1200',
    imageHeight: '630',
    imageAlt: 'Man - AI Platform',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Man - AI Platform',
    description:
      'Man — your personal AI. Train it on your own data, then chat, write and learn together.',
    image: '/og.png',
  },
} as const

/** Enables `env(safe-area-inset-*)` under notches / home indicators on mobile. */
export const viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
} as const
