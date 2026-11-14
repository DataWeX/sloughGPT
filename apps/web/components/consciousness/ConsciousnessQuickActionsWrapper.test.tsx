import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { ConsciousnessQuickActionsWrapper } from './ConsciousnessQuickActionsWrapper'
import { LocaleProvider } from '@/hooks/useLocale'

// Mutable so one test can walk the wrapper onto /chat and back.
const { mockPathname } = vi.hoisted(() => ({ mockPathname: { current: '/settings' } }))

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k, locale: 'en', setLocale: vi.fn() }),
  LocaleProvider: ({ children }: { children: React.ReactNode }) => children,
}))

vi.mock('@/lib/notify', () => ({
  notify: { toast: vi.fn(), banner: vi.fn() },

  useToastStore: Object.assign(
    vi.fn((sel: any) => sel({ addToast: vi.fn() })),
    { getState: () => ({ addToast: vi.fn() }) },
  ),
}))

vi.mock('@/vite/next-compat/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => mockPathname.current,
}))

function renderComponent() {
  return render(
    <LocaleProvider>
      <ConsciousnessQuickActionsWrapper />
    </LocaleProvider>,
  )
}

describe('ConsciousnessQuickActionsWrapper', () => {
  afterEach(() => {
    mockPathname.current = '/settings'
    cleanup()
  })

  it('renders the quick actions component', () => {
    renderComponent()
    expect(screen.getByLabelText('consciousness_quick_actions.toggle')).toBeTruthy()
  })

  it('opens panel when toggle is clicked', () => {
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_quick_actions.toggle'))
    expect(screen.getAllByText('consciousness_quick_actions.title').length).toBeGreaterThan(0)
  })

  it('does not mount the FAB on /chat (the toast viewport owns that corner)', () => {
    mockPathname.current = '/chat'
    const { container } = renderComponent()
    expect(container.firstChild).toBeNull()
    expect(screen.queryByLabelText('consciousness_quick_actions.toggle')).toBeNull()
  })

  it('does not mount the FAB on nested /chat routes either', () => {
    mockPathname.current = '/chat/abc123'
    const { container } = renderComponent()
    expect(container.firstChild).toBeNull()
  })
})
