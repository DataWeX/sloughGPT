import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import ConsciousnessAlertsPage from './page'
import { consciousnessController } from '@/lib/consciousness-controller'

const { mockAddToast } = vi.hoisted(() => ({
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    healthCheck: vi.fn(),
    getEpisodeHistory: vi.fn(),
    getStatus: vi.fn(),
    getSelfModel: vi.fn(),
    getQualia: vi.fn(),
    reflect: vi.fn(),
    updateConfig: vi.fn(),
    getTrainingStatus: vi.fn(),
    startTraining: vi.fn(),
    evaluate: vi.fn(),
    getQualiaHistory: vi.fn(),
    getBeliefsHistory: vi.fn(),
    submitFeedback: vi.fn(),
    seedData: vi.fn(),
    getPersonality: vi.fn(),
    updatePersonality: vi.fn(),
    resetPersonality: vi.fn(),
    clearEpisodes: vi.fn(),
    resetBeliefs: vi.fn(),
    getPersonalityHistory: vi.fn(),
    getPersonalityPresets: vi.fn(),
    applyPersonalityPreset: vi.fn(),
    getPersonalityConflicts: vi.fn(),
    listPersonas: vi.fn(),
    savePersona: vi.fn(),
    getPersona: vi.fn(),
    activatePersona: vi.fn(),
    deletePersona: vi.fn(),
    backup: vi.fn(),
    restore: vi.fn(),
    downloadBackup: vi.fn(),
    importBackup: vi.fn(),
    getStats: vi.fn(),
    batch: vi.fn(),
    connectStream: vi.fn(),
  },
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: vi.fn(() => ({
    t: (key: string) => key,
    locale: 'en',
  })),
  LocaleProvider: ({ children }: { children: React.ReactNode }) => children,
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: Object.assign(
    vi.fn((sel: any) => sel({ addToast: mockAddToast })),
    { getState: () => ({ addToast: mockAddToast }) },
  ),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

const mockHealth = {
  status: 'healthy',
  health_score: 85,
  enabled: true,
  level: 2,
  episodes: 50,
  avg_growth: 0.15,
  positive_ratio: 0.8,
  qualia: { valence: 0.5, arousal: 0.3, novelty: 0.6, coherence: 0.7, salience: 0.4, certainty: 0.3, complexity: 0.5 },
  last_reflection: 'System is performing well',
}

const mockEpisodes = {
  episodes: [
    { timestamp: Date.now() / 1000 - 100, growth_delta: 0.05, rating: 4, qualia: {} },
    { timestamp: Date.now() / 1000 - 50, growth_delta: 0.03, rating: 3, qualia: {} },
  ],
  total: 2,
}

describe('ConsciousnessAlertsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockAddToast.mockReset()
    vi.mocked(consciousnessController.healthCheck).mockResolvedValue(mockHealth as any)
    vi.mocked(consciousnessController.getEpisodeHistory).mockResolvedValue(mockEpisodes as any)
    localStorage.clear()
  })

  it('renders page title', async () => {
    render(<ConsciousnessAlertsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_alerts.page_title').length).toBeGreaterThan(0)
    })
  })

  it('shows alert cards section', async () => {
    render(<ConsciousnessAlertsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_alerts.active_alerts_title').length).toBeGreaterThan(0)
    })
  })

  it('shows alert rules section', async () => {
    render(<ConsciousnessAlertsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_alerts.rules_title').length).toBeGreaterThan(0)
    })
  })

  it('shows settings section', async () => {
    render(<ConsciousnessAlertsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_alerts.settings_title').length).toBeGreaterThan(0)
    })
  })

  it('calls healthCheck on mount', async () => {
    render(<ConsciousnessAlertsPage />)
    await waitFor(() => {
      expect(consciousnessController.healthCheck).toHaveBeenCalled()
    })
  })

  it('calls getEpisodeHistory on mount', async () => {
    render(<ConsciousnessAlertsPage />)
    await waitFor(() => {
      expect(consciousnessController.getEpisodeHistory).toHaveBeenCalledWith(50)
    })
  })

  it('shows active alerts count', async () => {
    vi.mocked(consciousnessController.healthCheck).mockResolvedValue({
      ...mockHealth,
      health_score: 10,
    } as any)
    render(<ConsciousnessAlertsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_alerts.active_alerts_title').length).toBeGreaterThan(0)
    })
    const countEl = screen.getAllByText('consciousness_alerts.active_alerts')
    expect(countEl.length).toBeGreaterThan(0)
  })

  it('shows rule enable/disable switches', async () => {
    render(<ConsciousnessAlertsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_alerts.rules_title').length).toBeGreaterThan(0)
    })
    const switches = screen.getAllByRole('switch')
    expect(switches.length).toBeGreaterThanOrEqual(5)
  })

  it('shows threshold inputs', async () => {
    render(<ConsciousnessAlertsPage />)
    await waitFor(() => {
      const inputs = screen.getAllByRole('spinbutton')
      expect(inputs.length).toBeGreaterThanOrEqual(5)
    })
  })

  it('shows settings toggles', async () => {
    render(<ConsciousnessAlertsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_alerts.notifications_label').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_alerts.sound_label').length).toBeGreaterThan(0)
    })
  })
})
