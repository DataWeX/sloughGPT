import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ConsciousnessSidebarWidget } from './ConsciousnessSidebarWidget'
import { LocaleProvider } from '@/hooks/useLocale'

vi.mock('@/hooks/useConsciousnessStatus', () => ({
  useConsciousnessStatus: vi.fn().mockReturnValue({
    status: {
      enabled: true,
      level: 2,
      episodes: 42,
      current_qualia: { valence: 0.5, arousal: 0.3, novelty: 0.6 },
    },
  }),
  getQualiaMood: vi.fn().mockReturnValue('Balanced'),
  getConsciousnessLevelLabel: vi.fn().mockReturnValue('Level 2'),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k, locale: 'en', setLocale: vi.fn() }),
  LocaleProvider: ({ children }: { children: React.ReactNode }) => children,
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconBrain: (p: any) => <span data-testid="icon-brain" {...p} />,
}))

function renderComponent() {
  return render(
    <LocaleProvider>
      <ConsciousnessSidebarWidget />
    </LocaleProvider>
  )
}

describe('ConsciousnessSidebarWidget', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the widget button', () => {
    renderComponent()
    expect(screen.getByLabelText('consciousness_sidebar.aria_label')).toBeTruthy()
  })

  it('shows level badge text', () => {
    renderComponent()
    expect(screen.getAllByText('consciousness_sidebar.level_badge').length).toBeGreaterThan(0)
  })

  it('shows brain icon', () => {
    renderComponent()
    expect(screen.getByTestId('icon-brain')).toBeTruthy()
  })

  it('shows enabled indicator', () => {
    renderComponent()
    const statusIndicator = screen.getAllByText('consciousness_sidebar.level_badge')
    expect(statusIndicator.length).toBeGreaterThan(0)
  })

  it('expands panel on click', () => {
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_sidebar.aria_label'))
    expect(screen.getAllByText('consciousness_sidebar.qualia_valence').length).toBeGreaterThan(0)
    expect(screen.getAllByText('consciousness_sidebar.qualia_arousal').length).toBeGreaterThan(0)
    expect(screen.getAllByText('consciousness_sidebar.qualia_novelty').length).toBeGreaterThan(0)
  })

  it('shows quick links when expanded', () => {
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_sidebar.aria_label'))
    expect(screen.getAllByText('consciousness_sidebar.link_dashboard').length).toBeGreaterThan(0)
    expect(screen.getAllByText('consciousness_sidebar.link_monitor').length).toBeGreaterThan(0)
    expect(screen.getAllByText('consciousness_sidebar.link_insights').length).toBeGreaterThan(0)
  })

  it('shows episode count when expanded', () => {
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_sidebar.aria_label'))
    expect(screen.getAllByText('consciousness_sidebar.episodes').length).toBeGreaterThan(0)
  })
})
