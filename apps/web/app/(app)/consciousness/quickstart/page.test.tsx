/**
 * Tests for the Consciousness Quickstart page.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessQuickstartPage from './page'

vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: vi.fn(),
  }),
}))

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    updateConfig: vi.fn(),
    applyPersonalityPreset: vi.fn(),
    seedData: vi.fn(),
  },
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({
    t: (key: string) => key,
    locale: 'en',
  }),
  LocaleProvider: ({ children }: { children: React.ReactNode }) => children,
}))

import { consciousnessController } from '@/lib/consciousness-controller'

const mockController = vi.mocked(consciousnessController)

const TestWrapper = ({ children }: { children: React.ReactNode }) => (
  <LocaleProvider>{children}</LocaleProvider>
)

describe('ConsciousnessQuickstartPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockController.updateConfig.mockResolvedValue({ updated: true })
    mockController.applyPersonalityPreset.mockResolvedValue({ applied: true })
    mockController.seedData.mockResolvedValue({ seeded: true })
    localStorage.clear()
  })

  it('renders the page', async () => {
    render(<ConsciousnessQuickstartPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_quickstart.page_title').length).toBeGreaterThan(0)
    })
  })

  it('displays welcome section', async () => {
    render(<ConsciousnessQuickstartPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_quickstart.welcome_title').length).toBeGreaterThan(0)
    })
  })

  it('displays benefits', async () => {
    render(<ConsciousnessQuickstartPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_quickstart.benefit1_title').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_quickstart.benefit2_title').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_quickstart.benefit3_title').length).toBeGreaterThan(0)
    })
  })

  it('displays steps section', async () => {
    render(<ConsciousnessQuickstartPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_quickstart.steps_title').length).toBeGreaterThan(0)
    })
  })

  it('displays all 5 steps', async () => {
    render(<ConsciousnessQuickstartPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_quickstart.step1_title').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_quickstart.step2_title').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_quickstart.step3_title').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_quickstart.step4_title').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_quickstart.step5_title').length).toBeGreaterThan(0)
    })
  })

  it('displays quick actions section', async () => {
    render(<ConsciousnessQuickstartPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_quickstart.quick_actions_title').length).toBeGreaterThan(0)
    })
  })

  it('displays enable button', async () => {
    render(<ConsciousnessQuickstartPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_quickstart.action_enable').length).toBeGreaterThan(0)
    })
  })

  it('displays personality button', async () => {
    render(<ConsciousnessQuickstartPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_quickstart.action_personality').length).toBeGreaterThan(0)
    })
  })

  it('displays seed button', async () => {
    render(<ConsciousnessQuickstartPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_quickstart.action_seed').length).toBeGreaterThan(0)
    })
  })

  it('displays progress section', async () => {
    render(<ConsciousnessQuickstartPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_quickstart.progress_title').length).toBeGreaterThan(0)
    })
  })

  it('displays progress percentage', async () => {
    render(<ConsciousnessQuickstartPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('0%').length).toBeGreaterThan(0)
    })
  })

  it('displays what\'s next section', async () => {
    render(<ConsciousnessQuickstartPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_quickstart.whats_next_title').length).toBeGreaterThan(0)
    })
  })

  it('displays navigation links', async () => {
    render(<ConsciousnessQuickstartPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('nav.consciousness_dashboard').length).toBeGreaterThan(0)
      expect(screen.getAllByText('nav.consciousness_training').length).toBeGreaterThan(0)
      expect(screen.getAllByText('nav.consciousness_history').length).toBeGreaterThan(0)
    })
  })
})
