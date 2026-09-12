/**
 * Tests for the Consciousness Settings page.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessSettingsPage from './page'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getStatus: vi.fn(),
    updateConfig: vi.fn(),
    seedData: vi.fn(),
    clearEpisodes: vi.fn(),
    resetBeliefs: vi.fn(),
    resetPersonality: vi.fn(),
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

const mockStatus = {
  enabled: true,
  level: 2,
  episodes: 50,
  beliefs: { helpful: 0.9, accurate: 0.7 },
  training: {
    is_training: false,
    total_pairs: 15,
  },
}

describe('ConsciousnessSettingsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockController.getStatus.mockResolvedValue(mockStatus as any)
    mockController.updateConfig.mockResolvedValue({ updated: true })
    mockController.seedData.mockResolvedValue({ seeded: true })
    mockController.clearEpisodes.mockResolvedValue({ cleared: true, episodes_cleared: 50 })
    mockController.resetBeliefs.mockResolvedValue({ reset: true, beliefs: {} })
    mockController.resetPersonality.mockResolvedValue({ reset: true })
  })

  it('renders the page', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_settings.page_title').length).toBeGreaterThan(0)
    })
  })

  it('displays config title', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_settings.config_title').length).toBeGreaterThan(0)
    })
  })

  it('displays enabled badge', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Active').length).toBeGreaterThan(0)
    })
  })

  it('displays level slider', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_settings.level_label').length).toBeGreaterThan(0)
    })
  })

  it('displays auto-reflect toggle', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_settings.auto_reflect_label').length).toBeGreaterThan(0)
    })
  })

  it('displays auto-evolve toggle', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_settings.auto_evolve_label').length).toBeGreaterThan(0)
    })
  })

  it('displays data section', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_settings.data_title').length).toBeGreaterThan(0)
    })
  })

  it('displays seed input', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_settings.seed_label').length).toBeGreaterThan(0)
    })
  })

  it('displays clear episodes button', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_settings.clear_episodes_label').length).toBeGreaterThan(0)
    })
  })

  it('displays clear beliefs button', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_settings.clear_beliefs_label').length).toBeGreaterThan(0)
    })
  })

  it('displays reset personality button', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_settings.reset_personality_label').length).toBeGreaterThan(0)
    })
  })

  it('displays display preferences section', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_settings.display_title').length).toBeGreaterThan(0)
    })
  })

  it('displays show in chat toggle', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_settings.show_in_chat_label').length).toBeGreaterThan(0)
    })
  })

  it('displays about section', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_settings.about_title').length).toBeGreaterThan(0)
    })
  })

  it('displays version', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('3.0.0').length).toBeGreaterThan(0)
    })
  })

  it('displays episodes count', async () => {
    render(<ConsciousnessSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('50').length).toBeGreaterThan(0)
    })
  })
})
