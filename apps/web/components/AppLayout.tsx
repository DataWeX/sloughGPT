'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'

import { cn, Button } from '@sloughgpt/strui'
import { IconMenu } from '@/components/icons/NavIcons'
import { Sidebar } from '@/components/Sidebar'
import { ErrorPanel } from '@sloughgpt/strui'
import { StatusBar } from '@/components/StatusBar'
import { OutputPanel } from '@/components/OutputPanel'
import { useApiMonitor } from '@/lib/api-monitor-store'
import { ToastContainer, RadixToastContainer } from '@/components/chat/Toast'
import { CommandPalette } from '@/components/CommandPalette'
import { useGlobalShortcuts } from '@/hooks/useGlobalShortcuts'
import { useToastStore } from '@/lib/toast-store'
import { KeyboardShortcutsModal } from '@/components/KeyboardShortcutsModal'
import { DebugOverlay } from '@/components/DebugOverlay'
import { WhatsNewDialog } from '@/components/WhatsNewDialog'
import { initLiveStatus } from '@/hooks/useLiveStatus'

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [portalMounted, setPortalMounted] = useState(false)
  const [showShortcuts, setShowShortcuts] = useState(false)
  const [showDebug, setShowDebug] = useState(false)
  const [showWhatsNew, setShowWhatsNew] = useState(false)
  const [showOutput, setShowOutput] = useState(false)
  const toasts = useToastStore(s => s.toasts)
  const dismissToast = useToastStore(s => s.dismissToast)
  const clearToasts = useToastStore(s => s.clearToasts)
  const apiStatus = useApiMonitor(s => s.status)
  useGlobalShortcuts()

  // Initialize live health SSE stream (replaces useBackendWatcher)
  useEffect(() => {
    return initLiveStatus()
  }, [])

  useEffect(() => {
    const handler = () => setShowShortcuts(true)
    window.addEventListener('toggle-shortcuts', handler)
    return () => window.removeEventListener('toggle-shortcuts', handler)
  }, [])

  useEffect(() => {
    const handler = () => setShowWhatsNew(true)
    window.addEventListener('toggle-whatsnew', handler)
    return () => window.removeEventListener('toggle-whatsnew', handler)
  }, [])

  useEffect(() => {
    const handler = () => setShowOutput(v => !v)
    window.addEventListener('toggle-output-panel', handler)
    return () => window.removeEventListener('toggle-output-panel', handler)
  }, [])

  useEffect(() => {
    setMobileNavOpen(false)
  }, [pathname])

  useEffect(() => {
    setPortalMounted(true)
  }, [])

  useEffect(() => {
    const mq = window.matchMedia('(min-width: 1024px)')
    const closeIfDesktop = () => {
      if (mq.matches) setMobileNavOpen(false)
    }
    closeIfDesktop()
    mq.addEventListener('change', closeIfDesktop)
    return () => mq.removeEventListener('change', closeIfDesktop)
  }, [])

  const closeMobileNav = () => setMobileNavOpen(false)

  return (
    <div className="flex h-dvh max-h-dvh flex-col bg-background pt-[env(safe-area-inset-top)]">
        {apiStatus === 'reloading' && (
          <div className="shrink-0 flex items-center justify-center gap-2 h-7 bg-warning/15 border-b border-warning/30 text-[11px] text-warning font-medium">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-warning animate-pulse" />
            Backend restarting — reconnecting…
          </div>
        )}
        <div className="flex min-h-0 flex-1 lg:flex-row">
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[200] focus:rounded-md focus:border focus:border-border focus:bg-card focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-foreground focus:shadow-lg focus:outline-none focus:ring-2 focus:ring-ring"
        >
          Skip to main content
        </a>

        <header className="flex shrink-0 items-center gap-2 border-b px-3 min-h-12 sm:min-h-14 lg:hidden">
          <Button
            variant="menu"
            size="icon"
            aria-expanded={mobileNavOpen}
            aria-controls="mobile-navigation-drawer"
            aria-haspopup="dialog"
            onClick={() => setMobileNavOpen((open) => !open)}
          >
            <IconMenu className="h-4 w-4" aria-hidden />
            <span className="sr-only">Open menu</span>
          </Button>
          <Link
            href="/"
            className="min-w-0 truncate text-sm font-semibold tracking-tight text-foreground hover:text-primary transition-colors"
          >
            sloughGPT
          </Link>
        </header>

        <div className="hidden h-full shrink-0 lg:flex">
          <Sidebar variant="desktop" />
        </div>

        <main
          id="main-content"
          className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden"
          tabIndex={-1}
        >
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden">{children}</div>
          <StatusBar />
        </main>
        </div>

        {portalMounted && createPortal(
          <>
            <div
              aria-hidden="true"
              className={cn(
                'fixed inset-0 z-[100] bg-black/45 backdrop-blur-sm lg:hidden transition-opacity duration-200',
                mobileNavOpen ? 'opacity-100' : 'opacity-0 pointer-events-none',
              )}
            />
            <div
              id="mobile-navigation-drawer"
              role="dialog"
              aria-modal="true"
              aria-label="Main navigation"
              className={cn(
                'fixed inset-y-0 left-0 z-[110] flex w-[min(var(--sidebar-width),min(18rem,92vw))] max-w-full outline-none lg:hidden',
                'shadow-[0.25rem_0_1.5rem_-0.25rem_rgba(0,0,0,0.28)]',
                'transition-all duration-200',
                mobileNavOpen ? 'translate-x-0' : '-translate-x-full pointer-events-none',
                'scrollbar-hide',
              )}
            >
              <span className="sr-only">Main navigation</span>
              <span className="sr-only">Primary navigation for the sloughGPT console. Choose a section or close this panel.</span>
              <Sidebar variant="drawer" onClose={closeMobileNav} onNavigate={closeMobileNav} />
            </div>
          </>,
          document.body,
        )}

        <ErrorPanel />
        <RadixToastContainer toasts={toasts} onDismiss={dismissToast} onClearAll={clearToasts} />
        <KeyboardShortcutsModal open={showShortcuts} onOpenChange={setShowShortcuts} />
        <DebugOverlay open={showDebug} onOpenChange={setShowDebug} />
        <CommandPalette />
        <WhatsNewDialog open={showWhatsNew} onOpenChange={setShowWhatsNew} />
        <OutputPanel open={showOutput} onClose={() => setShowOutput(false)} />
      </div>
  )
}
