'use client'

import * as DialogPrimitive from '@radix-ui/react-dialog'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { IconMenu } from '@/components/icons/NavIcons'
import { Sidebar } from '@/components/Sidebar'
import { ErrorPanel } from '@/components/ui/error-panel'
import { GlobalErrorHandler } from '@/components/GlobalErrorHandler'
import { StatusBar } from '@/components/StatusBar'
import { useApiMonitor } from '@/lib/api-monitor-store'
import { ToastContainer } from '@/components/chat/Toast'
import { useGlobalShortcuts } from '@/hooks/useGlobalShortcuts'
import { useBackendWatcher } from '@/hooks/useBackendWatcher'
import { useToastStore } from '@/lib/toast-store'
import { cn } from '@/lib/cn'
import { KeyboardShortcutsModal } from '@/components/KeyboardShortcutsModal'

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [showShortcuts, setShowShortcuts] = useState(false)
  const toasts = useToastStore(s => s.toasts)
  const dismissToast = useToastStore(s => s.dismissToast)
  const apiStatus = useApiMonitor(s => s.status)
  useGlobalShortcuts()
  useBackendWatcher()

  useEffect(() => {
    const handler = () => setShowShortcuts(true)
    window.addEventListener('toggle-shortcuts', handler)
    return () => window.removeEventListener('toggle-shortcuts', handler)
  }, [])

  useEffect(() => {
    setMobileNavOpen(false)
  }, [pathname])

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
    <DialogPrimitive.Root open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
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
          className="sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[200] focus:inline-block focus:rounded-md focus:border focus:border-border focus:bg-card focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-foreground focus:shadow-lg focus:outline-none focus:ring-2 focus:ring-ring"
        >
          Skip to main content
        </a>

        <header className="sl-mobile-header flex shrink-0 items-center gap-2 border-b px-3 min-h-12 sm:min-h-14 lg:hidden">
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
            Man
          </Link>
        </header>

        <div className="hidden h-full shrink-0 lg:flex">
          <Sidebar variant="desktop" />
        </div>

        <main
          id="main-content"
          className="sl-shell-main flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden"
          tabIndex={-1}
        >
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden">{children}</div>
          <StatusBar />
        </main>
        </div>

        <DialogPrimitive.Portal>
          <DialogPrimitive.Overlay
            className={cn(
              'fixed inset-0 z-[100] bg-black/45 backdrop-blur-sm lg:hidden',
              'data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 duration-200',
              // Match React state so taps pass through during close animation (don’t rely on data-state timing).
              !mobileNavOpen && 'pointer-events-none',
            )}
          />
          <DialogPrimitive.Content
            id="mobile-navigation-drawer"
            className={cn(
              'fixed inset-y-0 left-0 z-[110] flex w-[min(var(--sidebar-width),min(18rem,92vw))] max-w-full outline-none lg:hidden',
              'shadow-[0.25rem_0_1.5rem_-0.25rem_rgba(0,0,0,0.28)]',
              'data-[state=open]:animate-in data-[state=closed]:animate-out duration-200 ease-smooth',
              'data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left',
              !mobileNavOpen && 'pointer-events-none',
            )}
          >
            <DialogPrimitive.Title className="sr-only">Main navigation</DialogPrimitive.Title>
            <DialogPrimitive.Description className="sr-only">
              Primary navigation for the Man console. Choose a section or close this panel.
            </DialogPrimitive.Description>
            <Sidebar variant="drawer" onClose={closeMobileNav} onNavigate={closeMobileNav} />
          </DialogPrimitive.Content>
        </DialogPrimitive.Portal>

        <GlobalErrorHandler />
        <ErrorPanel />
        <ToastContainer toasts={toasts} onDismiss={dismissToast} />
        <KeyboardShortcutsModal open={showShortcuts} onOpenChange={setShowShortcuts} />
      </div>
    </DialogPrimitive.Root>
  )
}
