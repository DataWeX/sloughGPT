// @vitest-environment jsdom
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k, locale: 'en', setLocale: vi.fn() }),
  LocaleProvider: ({ children }: { children: React.ReactNode }) => children,
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: Object.assign(
    vi.fn((sel: any) => sel({ addToast: vi.fn() })),
    { getState: () => ({ addToast: vi.fn() }) },
  ),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

import { OverlayDock } from '@/components/overlay/OverlayDock'
import { useConsciousnessDockWidget } from './ConsciousnessQuickActions'
import { apiGet, apiPost } from '@/lib/http-client'

/** The hook through the real dock — the shape AppLayout renders. */
function Harness() {
  const widget = useConsciousnessDockWidget()
  return <OverlayDock widgets={[widget]} />
}

function renderComponent() {
  return render(<Harness />)
}

describe('ConsciousnessQuickActions (dock tenant)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })
  afterEach(cleanup)

  it('renders the M key on the rail', () => {
    renderComponent()
    expect(screen.getByLabelText('consciousness_quick_actions.toggle')).toBeTruthy()
    expect(screen.queryByText('consciousness_quick_actions.title')).toBeNull()
  })

  it('opens the panel on key click', () => {
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_quick_actions.toggle'))
    expect(screen.getAllByText('consciousness_quick_actions.title').length).toBeGreaterThan(0)
  })

  it('shows quick action buttons and nav links when open', () => {
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_quick_actions.toggle'))
    for (const key of ['reflect', 'seed', 'status', 'health'] as const) {
      expect(screen.getAllByText(`consciousness_quick_actions.${key}`).length).toBeGreaterThan(0)
    }
    expect(
      screen.getAllByText('consciousness_quick_actions.open_dashboard').length,
    ).toBeGreaterThan(0)
    expect(screen.getAllByText('consciousness_quick_actions.open_chat').length).toBeGreaterThan(0)
  })

  it('Ctrl+Shift+Q opens the panel through the dock event bridge', () => {
    renderComponent()
    expect(screen.queryByText('consciousness_quick_actions.title')).toBeNull()
    fireEvent.keyDown(window, { ctrlKey: true, shiftKey: true, key: 'Q' })
    expect(screen.getAllByText('consciousness_quick_actions.title').length).toBeGreaterThan(0)
  })

  it('runs an action through the API and clears on host outside-click', async () => {
    vi.mocked(apiPost).mockResolvedValue({ ok: true })
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_quick_actions.toggle'))
    fireEvent.click(screen.getByText('consciousness_quick_actions.reflect'))
    await waitFor(() => expect(apiPost).toHaveBeenCalledWith('/consciousness/reflect'))

    // Host owns outside-click dismiss (mousedown outside the dock region).
    fireEvent.mouseDown(document.body)
    await waitFor(() => expect(screen.queryByText('consciousness_quick_actions.title')).toBeNull())
  })

  it('status action hits GET', async () => {
    vi.mocked(apiGet).mockResolvedValue({ state: 'ok' })
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_quick_actions.toggle'))
    fireEvent.click(screen.getByText('consciousness_quick_actions.status'))
    await waitFor(() => expect(apiGet).toHaveBeenCalledWith('/consciousness/status'))
  })
})
