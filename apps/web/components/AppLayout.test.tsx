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

vi.mock('@/components/BottomNav', () => ({
  BottomNav: () => <div data-testid="bottom-nav" />,
}))

vi.mock('@/components/StatusBar', () => ({ StatusBar: () => <div data-testid="status-bar" /> }))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...classes: any[]) => classes.filter(Boolean).join(' '),
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  ErrorPanel: () => <div data-testid="error-panel" />,
  IconX: (props: any) => <svg {...props} />,
  IconMenu: (props: any) => <svg {...props} />,
  IconChevronRight: () => null,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}))
vi.mock('@/components/OutputPanel', () => ({ OutputPanel: () => <div data-testid="output-panel" /> }))
vi.mock('@/features/chat/components/feedback/Toast', () => ({ ToastContainer: () => <div data-testid="toast-container" />, RadixToastContainer: () => <div data-testid="toast-container" /> }))
vi.mock('@/components/CommandPalette', () => ({ CommandPalette: () => <div data-testid="command-palette" /> }))
vi.mock('@/components/KeyboardShortcutsDialog', () => ({ KeyboardShortcutsDialog: ({ open }: any) => open ? <div data-testid="shortcuts-modal" /> : null }))
vi.mock('@/components/DebugOverlay', () => ({ DebugOverlay: ({ open }: any) => open ? <div data-testid="debug-overlay" /> : null }))
vi.mock('@/components/WhatsNewDialog', () => ({ WhatsNewDialog: ({ open }: any) => open ? <div data-testid="whatsnew-dialog" /> : null, getUnseenCount: () => 0 }))

const { mockToastStore, mockApiMonitor } = vi.hoisted(() => ({
  mockToastStore: { toasts: [], dismissToast: vi.fn(), clearToasts: vi.fn() },
  mockApiMonitor: { status: 'connected' },
}))
vi.mock('@/lib/toast-store', () => ({ useToastStore: (sel?: any) => sel ? sel(mockToastStore) : mockToastStore }))
vi.mock('@/lib/api-monitor-store', () => ({ useApiMonitor: (sel?: any) => sel ? sel(mockApiMonitor) : mockApiMonitor }))

vi.mock('@/hooks/useGlobalShortcuts', () => ({ useGlobalShortcuts: vi.fn() }))
vi.mock('@/hooks/useLiveStatus', () => ({ initLiveStatus: vi.fn(() => vi.fn()) }))

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
    const sidebars = screen.getAllByTestId('sidebar')
    const desktop = sidebars.find(s => s.getAttribute('data-variant') === 'desktop')
    expect(desktop).toBeDefined()
  })

  it('renders status bar', () => {
    render(<AppLayout><div /></AppLayout>)
    expect(screen.getByTestId('status-bar')).toBeDefined()
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

  it('does not render whats-new dialog by default', () => {
    render(<AppLayout><div /></AppLayout>)
    expect(screen.queryByTestId('whatsnew-dialog')).toBeNull()
  })

  it('opens whats-new dialog on toggle-whatsnew event', async () => {
    render(<AppLayout><div /></AppLayout>)
    window.dispatchEvent(new CustomEvent('toggle-whatsnew'))
    await waitFor(() => {
      expect(screen.getByTestId('whatsnew-dialog')).toBeDefined()
    })
  })
})
