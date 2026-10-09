/**
 * Tests for the Consciousness Dashboard page.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessDashboardPage from './page'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getEpisodeHistory: vi.fn(),
    getQualiaHistory: vi.fn(),
    getBeliefsHistory: vi.fn(),
    evaluate: vi.fn(),
    getSelfModel: vi.fn(),
    getQualia: vi.fn(),
    getStatus: vi.fn(),
    seedData: vi.fn(),
    reflect: vi.fn(),
    submitFeedback: vi.fn(),
    updateConfig: vi.fn(),
  },
}))

vi.mock('@/hooks/useConsciousnessLive', () => ({
  useConsciousnessLive: () => ({
    isLive: false,
    lastUpdate: null,
    toggleLive: vi.fn(),
    latestEvent: null,
  }),
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

const mockEpisodes = [
  {
    timestamp: Date.now() / 1000 - 100,
    input_text: 'Test input 1',
    self_insight: 'Insight 1',
    growth_delta: 0.05,
    qualia: { valence: 0.5, arousal: 0.3, novelty: 0.6, coherence: 0.7 },
  },
  {
    timestamp: Date.now() / 1000 - 50,
    input_text: 'Test input 2',
    self_insight: 'Insight 2',
    growth_delta: -0.02,
    qualia: { valence: 0.4, arousal: 0.2, novelty: 0.5, coherence: 0.6 },
  },
]

const mockEvalReport = {
  overall_score: 75,
  diagnostics: ['Good coherence', 'Moderate growth'],
}

const mockStatus = {
  level: 1,
  enabled: true,
}

describe('ConsciousnessDashboardPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockController.getEpisodeHistory.mockResolvedValue({ episodes: mockEpisodes as any, total: 2 })
    mockController.getQualiaHistory.mockResolvedValue({ history: [] as any })
    mockController.getBeliefsHistory.mockResolvedValue({ beliefs: [] as any })
    mockController.evaluate.mockResolvedValue(mockEvalReport as any)
    mockController.getSelfModel.mockResolvedValue({ self_beliefs: { competence: 0.7, helpfulness: 0.8 } } as any)
    mockController.getQualia.mockResolvedValue({ valence: 0.5, arousal: 0.3 })
    mockController.getStatus.mockResolvedValue(mockStatus as any)
    mockController.seedData.mockResolvedValue({ seeded: true })
    mockController.reflect.mockResolvedValue({ reflection: 'Test reflection' })
    mockController.submitFeedback.mockResolvedValue({ accepted: true, belief_updates: [] })
    mockController.updateConfig.mockResolvedValue({ updated: true })
  })

  it('renders the page title', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Consciousness Dashboard').length).toBeGreaterThan(0)
    })
  })

  it('displays summary cards', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Total Episodes').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Quality Score').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Avg Growth').length).toBeGreaterThan(0)
    })
  })

  it('displays episodes count', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('2').length).toBeGreaterThan(0)
    })
  })

  it('displays eval report score', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('75/100').length).toBeGreaterThan(0)
    })
  })

  it('shows seed button', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Seed 30').length).toBeGreaterThan(0)
    })
  })

  it('shows reflect button', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Reflect').length).toBeGreaterThan(0)
    })
  })

  it('shows export button', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Export').length).toBeGreaterThan(0)
    })
  })

  it('shows consciousness level buttons', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Off').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Basic').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Full').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Deep').length).toBeGreaterThan(0)
    })
  })

  it('displays response quality section', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Response Quality').length).toBeGreaterThan(0)
    })
  })

  it('displays last reflection section', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Last Reflection').length).toBeGreaterThan(0)
    })
  })

  it('displays recent episodes section', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Recent Episodes').length).toBeGreaterThan(0)
    })
  })

  it('shows episode content', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Test input 1').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Test input 2').length).toBeGreaterThan(0)
    })
  })

  it('shows auto-refresh toggle', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_dashboard.autoRefresh').length).toBeGreaterThan(0)
    })
  })

  it('shows beliefs evolution chart section', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Beliefs Evolution').length).toBeGreaterThan(0)
    })
  })

  it('shows qualia history chart section', async () => {
    render(<ConsciousnessDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Qualia History').length).toBeGreaterThan(0)
    })
  })
})
