// @vitest-environment jsdom
/**
 * Journey: interrupted training job at boot -> process notification ->
 * user clicks Recover -> dispatch reaches the controller (which the
 * backend re-validates fail-closed).
 *
 * Card 20261102_014254. This is the end-to-end wiring test: AppLayout
 * mounts, loads the REST snapshot (/recovery/events — the boot source,
 * since SSE cannot deliver events fired before this client connected),
 * the dock renders the contract's action surface, and the click routes
 * through dispatchAction -> trainingController.recover. The server-authority
 * half (refusing a stale action) is pinned in
 * tests/server/test_dispatch_revalidation.py and the store's degradation
 * in NotificationDock.test.tsx — here we prove the happy path joins them.
 */

import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mockApiGet = vi.hoisted(() => vi.fn())
const mockRecover = vi.hoisted(() => vi.fn())

vi.mock('@/lib/http-client', () => ({
  apiGet: mockApiGet,
  apiPost: vi.fn(),
  apiPut: vi.fn(),
  apiDelete: vi.fn(),
  apiPatch: vi.fn(),
  authFetch: vi.fn(),
  streamSSE: vi.fn(),
  fetchSSE: vi.fn(),
}))

vi.mock('@/lib/training-controller', () => ({
  trainingController: {
    recover: mockRecover,
    stop: vi.fn(),
    recoverable: vi.fn().mockResolvedValue([]),
  },
}))

// ── AppLayout dependency scaffolding (mirrors AppLayout.test.tsx) ──────────
const mockPathname = '/training'
vi.mock('@/vite/next-compat/navigation', () => ({
  usePathname: () => mockPathname,
  useRouter: () => ({ push: vi.fn(), back: vi.fn(), forward: vi.fn(), refresh: vi.fn() }),
}))
vi.mock('@/vite/next-compat/link', () => ({
  default: ({ children, href, className }: any) => (
    <a href={href} className={className}>
      {children}
    </a>
  ),
}))
vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))
vi.mock('@/components/Sidebar', () => ({ Sidebar: () => <div data-testid="sidebar" /> }))
vi.mock('@/components/BottomNav', () => ({ BottomNav: () => <div data-testid="bottom-nav" /> }))
vi.mock('@/components/StatusBar', () => ({ StatusBar: () => <div data-testid="status-bar" /> }))
vi.mock('@/components/WorkspaceSwitcher', () => ({ WorkspaceSwitcher: () => null }))
vi.mock('@/components/OutputPanel', () => ({ OutputPanel: () => null }))
vi.mock('@/components/CommandPalette', () => ({ CommandPalette: () => null }))
vi.mock('@/components/KeyboardShortcutsDialog', () => ({ KeyboardShortcutsDialog: () => null }))
vi.mock('@/components/DebugOverlay', () => ({ DebugOverlay: () => null }))
vi.mock('@/components/WhatsNewDialog', () => ({ WhatsNewDialog: () => null }))
vi.mock('@/features/chat/components/feedback/Toast', () => ({
  RadixToastContainer: () => null,
}))
vi.mock('@/hooks/useGlobalShortcuts', () => ({ useGlobalShortcuts: vi.fn() }))
vi.mock('@/hooks/useConsciousnessShortcuts', () => ({ useConsciousnessShortcuts: vi.fn() }))
vi.mock('@/hooks/useLiveStatus', () => ({ initLiveStatus: vi.fn(() => vi.fn()) }))
vi.mock('@/lib/backend-retry', () => ({ probeBackendAndRevalidate: vi.fn() }))
vi.mock('@/lib/api-monitor-store', () => ({
  useApiMonitor: (sel?: any) => (sel ? sel({ status: 'connected' }) : { status: 'connected' }),
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: Object.assign((sel?: any) => (sel ? sel({ toasts: [] }) : { toasts: [] }), {
    getState: () => ({ toasts: [] }),
  }),
}))
vi.mock('@/lib/store', () => ({
  useAppStore: Object.assign((sel?: any) => (sel ? sel({ settings: {} }) : { settings: {} }), {
    getState: () => ({ settings: {} }),
  }),
}))

import AppLayout from './AppLayout'
import { clearProcessNotifications } from '@/lib/process-notifications'

const BOOT_SNAPSHOT = {
  contract_version: 1,
  count: 1,
  events: [
    {
      event_id: 'training:job-42:interrupted',
      kind: 'training',
      process_id: 'job-42',
      event: 'interrupted',
      state: 'interrupted',
      actions: [
        { id: 'recover_now', label: 'Recover now', intent: 'recover' },
        { id: 'dismiss', label: 'Dismiss', intent: 'dismiss' },
      ],
      process_link: '/training/job/job-42',
      title: 'Training interrupted',
      contract_version: 1,
      name: 'distill-run',
    },
  ],
}

beforeEach(() => {
  clearProcessNotifications()
  mockApiGet.mockReset()
  mockRecover.mockReset()
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  }))
})

afterEach(() => {
  clearProcessNotifications()
  vi.restoreAllMocks()
})

describe('journey: boot with interrupted job -> notify -> Recover', () => {
  it('shows the boot snapshot as a notification with its action surface', async () => {
    mockApiGet.mockResolvedValue(BOOT_SNAPSHOT)

    render(<AppLayout>content</AppLayout>)

    // Boot source hit:
    await waitFor(() => expect(mockApiGet).toHaveBeenCalledWith('/recovery/events'))
    // Notification rendered with the process's real commands:
    expect(await screen.findByText('Training interrupted')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Recover now' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /view job/i })).toHaveAttribute(
      'href',
      '/training/job/job-42',
    )
    expect(screen.getByText('distill-run')).toBeInTheDocument()
  })

  it('clicking Recover dispatches to the controller for that process', async () => {
    mockApiGet.mockResolvedValue(BOOT_SNAPSHOT)
    mockRecover.mockResolvedValue({ status: 'recovered' })

    render(<AppLayout>content</AppLayout>)

    const recoverBtn = await screen.findByRole('button', { name: 'Recover now' })
    fireEvent.click(recoverBtn)

    await waitFor(() => expect(mockRecover).toHaveBeenCalledWith('job-42'))
    // Dispatched confirmation renders (state change is server's to confirm).
    expect(await screen.findByText(/dispatched/i)).toBeInTheDocument()
  })

  it('an empty snapshot renders no dock at all', async () => {
    mockApiGet.mockResolvedValue({ contract_version: 1, count: 0, events: [] })

    render(<AppLayout>content</AppLayout>)

    await waitFor(() => expect(mockApiGet).toHaveBeenCalledWith('/recovery/events'))
    expect(screen.queryByLabelText('Process notifications')).not.toBeInTheDocument()
  })

  it('a snapshot outage never blocks the layout from rendering', async () => {
    mockApiGet.mockRejectedValue(new Error('backend down'))

    render(<AppLayout>content</AppLayout>)

    await waitFor(() => expect(mockApiGet).toHaveBeenCalled())
    expect(screen.getByTestId('status-bar')).toBeInTheDocument()
    expect(screen.queryByLabelText('Process notifications')).not.toBeInTheDocument()
  })
})
