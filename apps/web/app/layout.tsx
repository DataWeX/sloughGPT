import './globals.css'
import type { Metadata, Viewport } from 'next'
import localFont from 'next/font/local'

import { Providers } from './Providers'
import { MODE_STORAGE_KEY, THEME_IDS, THEME_STORAGE_KEY, PALETTE_IDS, PALETTE_STORAGE_KEY } from '@/lib/theme-storage'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import { ErrorLifecycle } from '@/components/ErrorLifecycle'
import WebVitals from '@/components/WebVitals'

const outfit = localFont({
  src: '../public/fonts/outfit-latin.woff2',
  variable: '--font-outfit',
  display: 'swap',
})

const jetbrainsMono = localFont({
  src: '../public/fonts/jetbrains-latin.woff2',
  variable: '--font-jetbrains-mono',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Man - AI Platform',
  description: 'Enterprise-grade AI framework with production-ready ML infrastructure',
  icons: { icon: '/favicon.svg' },
}

/** Enables `env(safe-area-inset-*)` under notches / home indicators on mobile. */
export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
}

/** Runs before React hydrates; ``useLayoutEffect`` in ThemeProvider re-syncs after any className reconciliation. */
const themeBootstrapInline = `!function(){try{var k=${JSON.stringify([...THEME_IDS])};var p=${JSON.stringify([...PALETTE_IDS])};var t=localStorage.getItem(${JSON.stringify(THEME_STORAGE_KEY)});var m=localStorage.getItem(${JSON.stringify(MODE_STORAGE_KEY)});var pl=localStorage.getItem(${JSON.stringify(PALETTE_STORAGE_KEY)});var r=document.documentElement;var th=k.indexOf(t)>=0?t:"purple";var mo="light"===m||"dark"===m?m:"dark";var pa=p.indexOf(pl)>=0?pl:"noir-violet";r.classList.remove("light","dark");k.forEach(function(id){r.classList.remove("theme-"+id)});p.forEach(function(id){r.classList.remove("palette-"+id)});r.classList.add(mo,"theme-"+th);if(pa!=="noir-violet"){r.classList.add("palette-"+pa);}}catch(e){}}();`
export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${outfit.variable} ${jetbrainsMono.variable}`} suppressHydrationWarning>
      <head>
      </head>
      <body className="font-sans antialiased" suppressHydrationWarning>
        <script dangerouslySetInnerHTML={{ __html: themeBootstrapInline }} />
        <WebVitals />
        <ErrorLifecycle />
        <ErrorBoundary>
          <Providers>{children}</Providers>
        </ErrorBoundary>
      </body>
    </html>
  )
}
