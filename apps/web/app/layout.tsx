import './globals.css'
import type { Metadata, Viewport } from 'next'
import { Rubik, Lato } from 'next/font/google'
import localFont from 'next/font/local'

import { Providers } from './Providers'
import { MODE_STORAGE_KEY, THEME_IDS, THEME_STORAGE_KEY, PALETTE_IDS, PALETTE_STORAGE_KEY } from '@/lib/theme-storage'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import { ErrorLifecycle } from '@/components/ErrorLifecycle'
import WebVitals from '@/components/WebVitals'

const rubik = Rubik({
  variable: '--font-rubik',
  display: 'swap',
  subsets: ['latin'],
})

const lato = Lato({
  weight: ['300', '400', '700', '900'],
  variable: '--font-lato',
  display: 'swap',
  subsets: ['latin'],
})

const jetbrainsMono = localFont({
  src: '../public/fonts/jetbrains-latin.woff2',
  variable: '--font-jetbrains-mono',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'SloughGPT',
  description: 'SloughGPT — AI chat interface',
  manifest: '/manifest.json',
  icons: {
    icon: '/favicon.svg',
    apple: '/favicon.svg',
  },
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'SloughGPT',
  },
  formatDetection: {
    telephone: false,
  },
  other: {
    'theme-color': '#7c52c4',
  },
}

/** Enables `env(safe-area-inset-*)` under notches / home indicators on mobile. */
export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: 'cover',
  themeColor: '#7c52c4',
}

/** Runs before React hydrates; ``useLayoutEffect`` in ThemeProvider re-syncs after any className reconciliation. */
const themeBootstrapInline = `!function(){try{var k=${JSON.stringify([...THEME_IDS])};var p=${JSON.stringify([...PALETTE_IDS])};var t=localStorage.getItem(${JSON.stringify(THEME_STORAGE_KEY)});var m=localStorage.getItem(${JSON.stringify(MODE_STORAGE_KEY)});var pl=localStorage.getItem(${JSON.stringify(PALETTE_STORAGE_KEY)});var r=document.documentElement;var th=k.indexOf(t)>=0?t:"purple";var mo="light"===m||"dark"===m?m:"dark";var pa=p.indexOf(pl)>=0?pl:"noir-violet";r.classList.remove("light","dark");k.forEach(function(id){r.classList.remove("theme-"+id)});p.forEach(function(id){r.classList.remove("palette-"+id)});r.classList.add(mo,"theme-"+th);if(pa!=="noir-violet"){r.classList.add("palette-"+pa);}}catch(e){}}();`
export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${rubik.variable} ${lato.variable} ${jetbrainsMono.variable}`} suppressHydrationWarning>
      <head>
        <link rel="manifest" href="/manifest.json" />
        <link rel="icon" href="/favicon.svg" />
        <link rel="apple-touch-icon" href="/favicon.svg" />
        <meta name="theme-color" content="#7c52c4" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="apple-mobile-web-app-title" content="SloughGPT" />
      </head>
      <body className="font-sans antialiased" suppressHydrationWarning>
        <script dangerouslySetInnerHTML={{ __html: themeBootstrapInline }} />
        <script
          dangerouslySetInnerHTML={{
            __html: `if('serviceWorker' in navigator){window.addEventListener('load',()=>{navigator.serviceWorker.register('/sw.js').catch(()=>{})});}`,
          }}
        />
        <WebVitals />
        <ErrorLifecycle />
        <ErrorBoundary>
          <Providers>{children}</Providers>
        </ErrorBoundary>
      </body>
    </html>
  )
}
