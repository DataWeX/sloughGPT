import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ConsciousnessQuickActions } from './ConsciousnessQuickActions'
import { LocaleProvider } from '@/hooks/useLocale'
import { apiGet, apiPost } from '@/lib/http-client'

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

function renderComponent() {
  return render(
    <LocaleProvider>
      <ConsciousnessQuickActions />
    </LocaleProvider>
  )
}

describe('ConsciousnessQuickActions', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders toggle button', () => {
    renderComponent()
    expect(screen.getByLabelText('consciousness_quick_actions.toggle')).toBeTruthy()
  })

  it('panel is hidden initially', () => {
    renderComponent()
    expect(screen.queryByText('consciousness_quick_actions.title')).toBeNull()
  })

  it('opens panel on click', () => {
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_quick_actions.toggle'))
    expect(screen.getAllByText('consciousness_quick_actions.title').length).toBeGreaterThan(0)
  })

  it('shows quick action buttons when open', () => {
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_quick_actions.toggle'))
    expect(screen.getAllByText('consciousness_quick_actions.reflect').length).toBeGreaterThan(0)
    expect(screen.getAllByText('consciousness_quick_actions.seed').length).toBeGreaterThan(0)
    expect(screen.getAllByText('consciousness_quick_actions.status').length).toBeGreaterThan(0)
    expect(screen.getAllByText('consciousness_quick_actions.health').length).toBeGreaterThan(0)
  })

  it('shows dashboard and chat links when open', () => {
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_quick_actions.toggle'))
    expect(screen.getAllByText('consciousness_quick_actions.open_dashboard').length).toBeGreaterThan(0)
    expect(screen.getAllByText('consciousness_quick_actions.open_chat').length).toBeGreaterThan(0)
  })

  it('closes panel on outside click', () => {
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_quick_actions.toggle'))
    expect(screen.getAllByText('consciousness_quick_actions.title').length).toBeGreaterThan(0)
    fireEvent.mouseDown(document.body)
    waitFor(() => {
      expect(screen.queryByText('consciousness_quick_actions.title')).toBeNull()
    })
  })
})
