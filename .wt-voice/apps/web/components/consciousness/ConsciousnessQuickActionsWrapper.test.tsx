import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { ConsciousnessQuickActionsWrapper } from './ConsciousnessQuickActionsWrapper'
import { LocaleProvider } from '@/hooks/useLocale'

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
      <ConsciousnessQuickActionsWrapper />
    </LocaleProvider>
  )
}

describe('ConsciousnessQuickActionsWrapper', () => {
  it('renders the quick actions component', () => {
    renderComponent()
    expect(screen.getByLabelText('consciousness_quick_actions.toggle')).toBeTruthy()
  })

  it('opens panel when toggle is clicked', () => {
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_quick_actions.toggle'))
    expect(screen.getAllByText('consciousness_quick_actions.title').length).toBeGreaterThan(0)
  })
})
