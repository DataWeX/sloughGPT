// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach, afterAll } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mockPathname = '/chat'
vi.mock('next/navigation', () => ({
  usePathname: () => mockPathname,
}))

vi.mock('next/link', () => ({ default: ({ children, href, className }: any) => <a href={href} className={className}>{children}</a> }))

vi.mock('@/components/Sidebar', () => ({
  Sidebar: vi.fn().mockImplementation(({ variant, onClose, onNavigate }: any) => (
    <div data-testid="sidebar" data-variant={variant}>
      {onClose && <button data-testid="sidebar-close" onClick={onClose}>Close</button>}
      {onNavigate && <button data-testid="sidebar-navigate" onClick={onNavigate}>Navigate</button>}
    </div>
  )),
}))

vi.mock('@/components/StatusBar', () => ({ StatusBar: () => <div data-testid="status-bar" /> }))
vi.mock('@/components/GlobalErrorHandler', () => ({ GlobalErrorHandler: () => <div data-testid="error-handler" /> }))
vi.mock('@/components/ui/error-panel', () => ({ ErrorPanel: () => <div data-testid="error-panel" /> }))
vi.mock('@/components/chat/Toast', () => ({ ToastContainer: () => <div data-testid="toast-container" /> }))
vi.mock('@/components/CommandPalette', () => ({ CommandPalette: () => <div data-testid="command-palette" /> }))
vi.mock('@/components/KeyboardShortcutsModal', () => ({ KeyboardShortcutsModal: ({ open }: any) => open ? <div data-testid="shortcuts-modal" /> : null }))
vi.mock('@/components/DebugOverlay', () => ({ DebugOverlay: ({ open }: any) => open ? <div data-testid="debug-overlay" /> : null }))

const { mockToastStore, mockApiMonitor } = vi.hoisted(() => ({
  mockToastStore: { toasts: [], dismissToast: vi.fn(), clearToasts: vi.fn() },
  mockApiMonitor: { status: 'connected' },
}))
vi.mock('@/lib/toast-store', () => ({ useToastStore: (sel?: any) => sel ? sel(mockToastStore) : mockToastStore }))
vi.mock('@/lib/api-monitor-store', () => ({ useApiMonitor: (sel?: any) => sel ? sel(mockApiMonitor) : mockApiMonitor }))

vi.mock('@/hooks/useGlobalShortcuts', () => ({ useGlobalShortcuts: vi.fn() }))
vi.mock('@/hooks/useBackendWatcher', () => ({ useBackendWatcher: vi.fn() }))

import AppLayout from './AppLayout'

describe('AppLayout', () => {
  const origMatchMedia = window.matchMedia

  beforeEach(() => {
    window.matchMedia = vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }))
  })

  afterEach(() => {
    window.matchMedia = origMatchMedia
    cleanup()
  })

  it('renders children', () => {
    render(<AppLayout><div data-testid="child">Hello</div></AppLayout>)
    expect(screen.getByTestId('child')).toBeDefined()
  })

  it('renders desktop sidebar', () => {
    render(<AppLayout><div /></AppLayout>)
    const sidebar = screen.getByTestId('sidebar')
    expect(sidebar).toBeDefined()
    expect(sidebar.getAttribute('data-variant')).toBe('desktop')
  })

  it('renders status bar', () => {
    render(<AppLayout><div /></AppLayout>)
    expect(screen.getByTestId('status-bar')).toBeDefined()
  })

  it('renders GlobalErrorHandler', () => {
    render(<AppLayout><div /></AppLayout>)
    expect(screen.getByTestId('error-handler')).toBeDefined()
  })

  it('renders error panel', () => {
    render(<AppLayout><div /></AppLayout>)
    expect(screen.getByTestId('error-panel')).toBeDefined()
  })

  it('renders toast container', () => {
    render(<AppLayout><div /></AppLayout>)
    expect(screen.getByTestId('toast-container')).toBeDefined()
  })

  it('renders command palette', () => {
    render(<AppLayout><div /></AppLayout>)
    expect(screen.getByTestId('command-palette')).toBeDefined()
  })

  it('has skip-to-main-content link for accessibility', () => {
    render(<AppLayout><div /></AppLayout>)
    const skipLink = screen.getByText('Skip to main content')
    expect(skipLink).toBeDefined()
    expect(skipLink.getAttribute('href')).toBe('#main-content')
  })

  it('has main content area with id', () => {
    render(<AppLayout><div /></AppLayout>)
    const main = document.getElementById('main-content')
    expect(main).toBeDefined()
  })

  it('shows reloading banner when apiStatus is reloading', () => {
    mockApiMonitor.status = 'reloading'
    render(<AppLayout><div /></AppLayout>)
    expect(screen.getByText('Backend restarting — reconnecting…')).toBeDefined()
    mockApiMonitor.status = 'connected'
  })

  it('closes mobile nav on pathname change', () => {
    render(<AppLayout><div /></AppLayout>)
    expect(screen.queryByTestId('sidebar-close')).toBeDefined()
  })

  it('renders mobile header with hamburger', () => {
    render(<AppLayout><div /></AppLayout>)
    expect(screen.getByText('sloughGPT')).toBeDefined()
  })
})
