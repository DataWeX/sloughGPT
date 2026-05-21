import './globals.css'
import type { Metadata, Viewport } from 'next'
import { JetBrains_Mono, Outfit } from 'next/font/google'

import { Providers } from './Providers'
import { MODE_STORAGE_KEY, THEME_IDS, THEME_STORAGE_KEY } from '@/lib/theme-storage'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import { SuppressDevOverlay } from '@/components/SuppressDevOverlay'

const outfit = Outfit({
  subsets: ['latin'],
  variable: '--font-outfit',
  display: 'swap',
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'SloughGPT - Enterprise AI',
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
const themeBootstrapInline = `!function(){try{var k=${JSON.stringify([...THEME_IDS])};var t=localStorage.getItem(${JSON.stringify(THEME_STORAGE_KEY)});var m=localStorage.getItem(${JSON.stringify(MODE_STORAGE_KEY)});var r=document.documentElement;var th=k.indexOf(t)>=0?t:"purple";var mo="light"===m||"dark"===m?m:"dark";r.classList.remove("light","dark");k.forEach(function(id){r.classList.remove("theme-"+id)});r.classList.add(mo,"theme-"+th);}catch(e){}}();`
export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${outfit.variable} ${jetbrainsMono.variable}`} suppressHydrationWarning>
      <body className="font-sans antialiased">
        <script dangerouslySetInnerHTML={{ __html: themeBootstrapInline }} />
        <SuppressDevOverlay />
        <ErrorBoundary>
          <Providers>{children}</Providers>
        </ErrorBoundary>
      </body>
    </html>
  )
}
