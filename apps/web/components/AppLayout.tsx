'use client'

import Link from '@/vite/next-compat/link'
import { usePathname } from '@/vite/next-compat/navigation'
import { useEffect, useRef, useState } from 'react'
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
import { GlobalBanner } from '@/components/GlobalBanner'
import { useGlobalShortcuts } from '@/hooks/useGlobalShortcuts'
import { useConsciousnessShortcuts } from '@/hooks/useConsciousnessShortcuts'
import { useToastStore } from '@/lib/toast-store'
import { useAppStore } from '@/lib/store'
import { useBannerStore } from '@/lib/banner-store'
import { KeyboardShortcutsDialog } from '@/components/KeyboardShortcutsDialog'
import { DebugOverlay } from '@/components/DebugOverlay'
import { WhatsNewDialog } from '@/components/WhatsNewDialog'
import { initLiveStatus } from '@/hooks/useLiveStatus'
import { probeBackendAndRevalidate } from '@/lib/backend-retry'
import { ConvSidebarProvider, useConvSidebar } from '@/features/chat/contexts/ConvSidebarContext'
import { WorkspaceSwitcher } from '@/components/WorkspaceSwitcher'
import { useLocale } from '@/hooks/useLocale'

/** Hold-off before a non-connected status raises the global drop banner —
 *  one failed poll is a blip, not an outage the user should be told about.
 *  Default only: the live value is the `bannerGraceMs` setting (see below). */
export const OFFLINE_BANNER_GRACE_MS = 5_000

/** How long the green "Backend reconnected" confirmation stays up before it
 *  retires itself — long enough to read, short enough to be gone.
 *  Default only: the live value is the `bannerHoldMs` setting (0 = no
 *  confirmation). */
export const RECOVERY_BANNER_HOLD_MS = 4_000

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <ConvSidebarProvider>
      <AppLayoutInner>{children}</AppLayoutInner>
    </ConvSidebarProvider>
  )
}

function AppLayoutInner({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const { t } = useLocale()
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [portalMounted, setPortalMounted] = useState(false)
  const [showShortcuts, setShowShortcuts] = useState(false)
  const [showDebug, setShowDebug] = useState(false)
  const [showWhatsNew, setShowWhatsNew] = useState(false)
  const [showOutput, setShowOutput] = useState(false)
  const toasts = useToastStore((s) => s.toasts)
  const dismissToast = useToastStore((s) => s.dismissToast)
  const clearToasts = useToastStore((s) => s.clearToasts)
  const apiStatus = useApiMonitor((s) => s.status)
  const { open: convOpen, navCollapsed, toggleNav, setNavCollapsed, toggleConv } = useConvSidebar()
  useGlobalShortcuts()
  useConsciousnessShortcuts()

  // Initialize live health SSE stream
  useEffect(() => {
    return initLiveStatus()
  }, [])

  useEffect(() => {
    const handler = () => setShowShortcuts((v) => !v)
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
    const handler = () => setShowWhatsNew((v) => !v)
    window.addEventListener('toggle-whatsnew', handler)
    return () => window.removeEventListener('toggle-whatsnew', handler)
  }, [])

  useEffect(() => {
    const handler = () => setShowOutput((v) => !v)
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

  // Backend health lives on ONE banner surface (GlobalBanner + useBannerStore).
  // Lifecycle mirrors the CONNECTION, not the request:
  //   offline/error → grace → red "unreachable" (held as long as the drop)
  //   connected     → green "reconnected" confirmation → auto-dismiss
  // The ref records that a drop was actually shown, so (a) booting online
  // never flashes a green banner, and (b) recovery only confirms what the
  // user saw. Same key 'backend-connection' throughout — red and green
  // replace each other in the single slot instead of stacking two rows.
  // Timings are settings-backed flags (bannerGraceMs / bannerHoldMs in
  // lib/store): read via getState() when each timer arms, so editing a flag
  // applies to the NEXT lifecycle event without re-running this effect and
  // re-arming an in-flight grace/hold. Local settings — readable offline,
  // which is exactly when this banner is up.
  const connectionDropRef = useRef(false)
  useEffect(() => {
    const down = apiStatus === 'offline' || apiStatus === 'error'
    if (down) {
      const { bannerGraceMs } = useAppStore.getState().settings
      let id: string | null = null
      const raise = () => {
        id = useBannerStore.getState().showBanner({
          key: 'backend-connection',
          tone: 'destructive',
          title: 'Backend unreachable',
          message: 'The connection dropped — retrying automatically.',
          action: {
            label: 'Retry',
            onAction: () => {
              // GlobalBanner retires the row on action-click, and a FAILED
              // probe changes no status — so without this the user's Retry
              // would silently swallow the outage notice while the backend
              // is still down. Re-raise unless another producer took the
              // slot meanwhile (never displace an unrelated banner).
              void probeBackendAndRevalidate().then((ok) => {
                if (
                  !ok &&
                  !useBannerStore.getState().banners.some((b) => b.key !== 'backend-connection')
                ) {
                  raise()
                }
              })
            },
          },
        })
        connectionDropRef.current = true
      }
      const timer = setTimeout(raise, bannerGraceMs ?? OFFLINE_BANNER_GRACE_MS)
      return () => {
        clearTimeout(timer)
        // A superseded id is a no-op (single-slot store) — another producer's
        // banner is left alone. `id` tracks the latest raise, so a re-raised
        // banner is retired here too when the status moves on.
        if (id) useBannerStore.getState().dismissBanner(id)
      }
    }

    // Recovery confirmation: only after a drop the user actually saw, and only
    // once the connection is genuinely back — 'connecting'/'reloading' keep
    // waiting for 'connected' before confirming.
    if (apiStatus !== 'connected' || !connectionDropRef.current) return
    // Never displace an unrelated banner that took the slot meanwhile.
    const current = useBannerStore.getState().banners[0]
    if (current && current.key !== 'backend-connection') {
      connectionDropRef.current = false
      return
    }
    const { bannerHoldMs } = useAppStore.getState().settings
    // 0 disables the recovery confirmation — the drop ref clears silently.
    if ((bannerHoldMs ?? RECOVERY_BANNER_HOLD_MS) <= 0) {
      connectionDropRef.current = false
      return
    }
    const greenId = useBannerStore.getState().showBanner({
      key: 'backend-connection',
      tone: 'success',
      title: 'Backend reconnected',
      message: 'The connection is stable again.',
    })
    // The drop ref clears only when the hold actually completes, so a
    // StrictMode remount (cleanup clears the timer, ref still set) re-raises
    // the banner and re-arms the hold instead of leaving it stuck on screen.
    const timer = setTimeout(() => {
      useBannerStore.getState().dismissBanner(greenId)
      connectionDropRef.current = false
    }, bannerHoldMs ?? RECOVERY_BANNER_HOLD_MS)
    return () => {
      clearTimeout(timer)
      // Retire the green banner on re-run/unmount; the drop ref stays set so a
      // StrictMode second pass re-shows it (single slot replaces, never stacks).
      useBannerStore.getState().dismissBanner(greenId)
    }
  }, [apiStatus])

  return (
    <div className="sl-app-shell">
      {/* Global banner system — one banner surface for all pages. The backend
          "restarting" notice is raised through the store (see effect above), so
          it never stacks a second row beside the boot-stall banner. */}
      <GlobalBanner />

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
        <WorkspaceSwitcher />
        <Link
          href="/"
          className="min-w-0 truncate text-[11px] font-semibold tracking-tight text-foreground hover:text-primary transition-colors"
        >
          {t('app.name')}
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
        <div
          className="sl-app-sidebar-desktop relative"
          data-collapsed={navCollapsed ? 'true' : undefined}
        >
          <Sidebar variant="desktop" collapsed={navCollapsed} onToggleCollapse={toggleNav} />
          {/* 3D bookmark tab — protrudes from sidebar edge, floats OVER the
              conversation panel like a sticker/tray. z-40 keeps it above the
              panel (which reserves a matching gutter via --nav-tab-protrusion)
              so it stays hoverable/clickable at any scroll position. */}
          <button
            type="button"
            onClick={toggleNav}
            aria-label={navCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={navCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className={cn(
              'group/tab absolute top-1/2 z-40 flex h-[4.5rem] w-[var(--nav-tab-protrusion)] -translate-y-1/2 items-center justify-center',
              'rounded-r-md cursor-pointer',
              'transition-all duration-300 ease-[cubic-bezier(0.222,0.133,0,1)]',
              'hover:w-[calc(var(--nav-tab-protrusion)_+_0.25rem)]',
              'right-0 translate-x-[calc(100%-1px)]',
              'bg-gradient-to-b from-primary/80 via-primary to-primary/90',
              'shadow-[1px_0_2px_-1px_rgba(0,0,0,0.2),2px_0_4px_-2px_rgba(0,0,0,0.15),3px_0_8px_-3px_rgba(0,0,0,0.1)]',
              'before:pointer-events-none before:absolute before:inset-x-0 before:top-0 before:h-px before:rounded-r-md before:bg-gradient-to-b before:from-white/40 before:to-transparent',
              'after:pointer-events-none after:absolute after:inset-x-0 after:bottom-0 after:h-px after:rounded-r-md after:bg-gradient-to-t after:from-black/20 after:to-transparent',
              'hover:shadow-[1px_0_2px_-1px_rgba(0,0,0,0.2),2px_0_4px_-2px_rgba(0,0,0,0.15),3px_0_8px_-3px_rgba(0,0,0,0.1),0_0_12px_-2px_rgba(var(--primary)/0.3)]',
            )}
          >
            <IconChevronRight
              className={cn(
                'h-3 w-3 text-primary-foreground transition-transform duration-300',
                'drop-shadow-[0_1px_1px_rgba(0,0,0,0.3)]',
                navCollapsed ? 'rotate-0' : 'rotate-180',
                'group-hover/tab:scale-110',
              )}
            />
          </button>
        </div>

        {/* Main content area */}
        <main id="main-content" className="sl-app-main" tabIndex={-1}>
          <div className="sl-app-content">{children}</div>
          <StatusBar />
        </main>
      </div>

      {/* Mobile bottom navigation */}
      <BottomNav />

      {/* Mobile drawer portal */}
      {portalMounted &&
        createPortal(
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
              <span className="sr-only">
                Primary navigation for the Man console. Choose a section or close this panel.
              </span>
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
