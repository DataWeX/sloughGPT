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
import { ConvSidebarProvider, useConvSidebar } from '@/contexts/ConvSidebarContext'

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <ConvSidebarProvider>
      <AppLayoutInner>{children}</AppLayoutInner>
    </ConvSidebarProvider>
  )
}

function AppLayoutInner({ children }: { children: React.ReactNode }) {
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
  const { open: convOpen, navCollapsed, toggleNav, setNavCollapsed, toggleConv } = useConvSidebar()
  useGlobalShortcuts()

  // Initialize live health SSE stream
  useEffect(() => {
    return initLiveStatus()
  }, [])

  useEffect(() => {
    const handler = () => setShowShortcuts(true)
    window.addEventListener('toggle-shortcuts', handler)
    return () => window.removeEventListener('toggle-shortcuts', handler)
  }, [])

  useEffect(() => {
    const handler = () => toggleNav()
    window.addEventListener('toggle-nav-sidebar', handler)
    return () => window.removeEventListener('toggle-nav-sidebar', handler)
  }, [toggleNav])

  useEffect(() => {
    const handler = () => toggleConv()
    window.addEventListener('toggle-conv-sidebar', handler)
    return () => window.removeEventListener('toggle-conv-sidebar', handler)
  }, [toggleConv])

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
    <div className="sl-app-shell">
      {/* Restarting banner */}
      {apiStatus === 'reloading' && (
        <div className="sl-app-banner">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-warning animate-pulse" />
          Backend restarting — reconnecting…
        </div>
      )}

      {/* Main horizontal split: sidebar + content */}
      <div className="sl-app-body">
        {/* Skip link */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[200] focus:rounded-md focus:border focus:border-border focus:bg-card focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-foreground focus:shadow-lg focus:outline-none focus:ring-2 focus:ring-ring"
        >
          Skip to main content
        </a>

        {/* Mobile header */}
        <header className="sl-app-mobile-header">
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

        {/* Desktop sidebar */}
        <div className="sl-app-sidebar-desktop" data-collapsed={navCollapsed ? 'true' : undefined}>
          <Sidebar variant="desktop" collapsed={navCollapsed} onToggleCollapse={toggleNav} />
        </div>

        {/* Main content area */}
        <main
          id="main-content"
          className="sl-app-main"
          tabIndex={-1}
        >
          <div className="sl-app-content">{children}</div>
          <StatusBar />
        </main>
      </div>

      {/* Mobile drawer portal */}
      {portalMounted && createPortal(
        <>
          <div
            aria-hidden="true"
            className={cn(
              'sl-app-drawer-backdrop',
              mobileNavOpen ? 'opacity-100' : 'opacity-0 pointer-events-none',
            )}
          />
          <div
            id="mobile-navigation-drawer"
            role="dialog"
            aria-modal="true"
            aria-label="Main navigation"
            className={cn(
              'sl-app-drawer',
              mobileNavOpen ? 'translate-x-0' : '-translate-x-full pointer-events-none',
            )}
          >
            <span className="sr-only">Main navigation</span>
            <span className="sr-only">Primary navigation for the sloughGPT console. Choose a section or close this panel.</span>
            <Sidebar variant="drawer" onClose={closeMobileNav} onNavigate={closeMobileNav} />
          </div>
        </>,
        document.body,
      )}

      {/* Overlays */}
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
