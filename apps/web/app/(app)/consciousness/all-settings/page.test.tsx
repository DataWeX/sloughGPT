/**
 * Tests for the Consciousness All Settings page.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { LocaleProvider } from '@/hooks/useLocale'
import AllSettingsPage from './page'

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
  training: { is_training: false, total_pairs: 15 },
}

describe('AllSettingsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockController.getStatus.mockResolvedValue(mockStatus as any)
    mockController.updateConfig.mockResolvedValue({ updated: true })
    mockController.seedData.mockResolvedValue({ seeded: true })
    mockController.clearEpisodes.mockResolvedValue({ cleared: true, episodes_cleared: 50 })
    mockController.resetBeliefs.mockResolvedValue({ reset: true, beliefs: {} })
    mockController.resetPersonality.mockResolvedValue({ reset: true })
    localStorage.clear()
  })

  it('renders the page', async () => {
    render(<AllSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_all_settings.page_title').length).toBeGreaterThan(0)
    })
  })

  it('displays tab list', async () => {
    render(<AllSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_all_settings.tab_core').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_all_settings.tab_display').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_all_settings.tab_personality').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_all_settings.tab_data').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_all_settings.tab_notifications').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_all_settings.tab_export').length).toBeGreaterThan(0)
    })
  })

  it('displays core settings by default', async () => {
    render(<AllSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_all_settings.core_title').length).toBeGreaterThan(0)
    })
  })

  it('displays level label', async () => {
    render(<AllSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_all_settings.level_label').length).toBeGreaterThan(0)
    })
  })

  it('displays enabled badge', async () => {
    render(<AllSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Active').length).toBeGreaterThan(0)
    })
  })

  it('displays auto-reflect toggle', async () => {
    render(<AllSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_all_settings.auto_reflect_label').length).toBeGreaterThan(0)
    })
  })

  it('displays auto-evolve toggle', async () => {
    render(<AllSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_all_settings.auto_evolve_label').length).toBeGreaterThan(0)
    })
  })

  it('displays level descriptions', async () => {
    render(<AllSettingsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Full self-model with belief tracking').length).toBeGreaterThan(0)
    })
  })
})
