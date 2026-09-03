'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'

import { cn, Button, IconChevronRight } from '@sloughgpt/strui'
import { IconMenu } from '@/components/icons/NavIcons'
import { Sidebar } from '@/components/Sidebar'
import { BottomNav } from '@/components/BottomNav'
import { ErrorPanel } from '@sloughgpt/strui'
import { StatusBar } from '@/components/StatusBar'
import { OutputPanel } from '@/components/OutputPanel'
import { useApiMonitor } from '@/lib/api-monitor-store'
import { RadixToastContainer } from '@/features/chat/components/feedback/Toast'
import { CommandPalette } from '@/components/CommandPalette'
import { useGlobalShortcuts } from '@/hooks/useGlobalShortcuts'
import { useToastStore } from '@/lib/toast-store'
import { KeyboardShortcutsDialog } from '@/components/KeyboardShortcutsDialog'
import { DebugOverlay } from '@/components/DebugOverlay'
import { WhatsNewDialog } from '@/components/WhatsNewDialog'
import { initLiveStatus } from '@/hooks/useLiveStatus'
import { ConvSidebarProvider, useConvSidebar } from '@/features/chat/contexts/ConvSidebarContext'

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

  // Global toast listener — bridges show-toast custom events to the toast store
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail
      if (detail?.message) {
        useToastStore.getState().addToast(detail.message, detail.type || 'info')
      }
    }
    window.addEventListener('show-toast', handler)
    return () => window.removeEventListener('show-toast', handler)
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

      {/* Mobile header — full-width top bar on < lg */}
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

      {/* Main horizontal split: sidebar + content */}
      <div className="sl-app-body">
        {/* Skip link */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[200] focus:rounded-md focus:border focus:border-border focus:bg-card focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-foreground focus:shadow-lg focus:outline-none focus:ring-2 focus:ring-ring"
        >
          Skip to main content
        </a>

        {/* Desktop sidebar */}
        <div className="sl-app-sidebar-desktop relative" data-collapsed={navCollapsed ? 'true' : undefined}>
          <Sidebar variant="desktop" collapsed={navCollapsed} onToggleCollapse={toggleNav} />
          {/* 3D bookmark tab — protrudes from sidebar edge */}
          <button
            type="button"
            onClick={toggleNav}
            aria-label={navCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={navCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className={cn(
              'group/tab absolute top-1/2 z-20 flex h-[4.5rem] w-[1.15rem] -translate-y-1/2 items-center justify-center',
              'rounded-r-md cursor-pointer',
              'transition-all duration-300 ease-[cubic-bezier(222,133,0,1)\_]',
              'hover:w-[1.4rem]',
              'right-0 translate-x-[calc(100%-1px)]',
              'bg-gradient-to-b from-primary/80 via-primary to-primary/90',
              'shadow-[1px_0_2px_-1px_rgba(0,0,0,0.2),2px_0_4px_-2px_rgba(0,0,0,0.15),3px_0_8px_-3px_rgba(0,0,0,0.1)]',
              'before:pointer-events-none before:absolute before:inset-x-0 before:top-0 before:h-px before:rounded-r-md before:bg-gradient-to-b before:from-white/40 before:to-transparent',
              'after:pointer-events-none after:absolute after:inset-x-0 after:bottom-0 after:h-px after:rounded-r-md after:bg-gradient-to-t after:from-black/20 after:to-transparent',
              'hover:shadow-[1px_0_2px_-1px_rgba(0,0,0,0.2),2px_0_4px_-2px_rgba(0,0,0,0.15),3px_0_8px_-3px_rgba(0,0,0,0.1),0_0_12px_-2px_rgba(var(--primary)/0.3)]',
            )}
          >
            <IconChevronRight className={cn(
              'h-3 w-3 text-primary-foreground transition-transform duration-300',
              'drop-shadow-[0_1px_1px_rgba(0,0,0,0.3)]',
              navCollapsed ? 'rotate-0' : 'rotate-180',
              'group-hover/tab:scale-110',
            )} />
          </button>
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

      {/* Mobile bottom navigation */}
      <BottomNav />

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
      <KeyboardShortcutsDialog open={showShortcuts} onOpenChange={setShowShortcuts} />
      <DebugOverlay open={showDebug} onOpenChange={setShowDebug} />
      <CommandPalette />
      <WhatsNewDialog open={showWhatsNew} onOpenChange={setShowWhatsNew} />
      <OutputPanel open={showOutput} onClose={() => setShowOutput(false)} />
    </div>
  )
}
