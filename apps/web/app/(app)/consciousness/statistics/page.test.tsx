import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessStatisticsPage from './page'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getEpisodeHistory: vi.fn(),
    getStatus: vi.fn(),
    healthCheck: vi.fn(),
    evaluate: vi.fn(),
  },
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({
    t: (key: string) => key,
    locale: 'en',
  }),
  LocaleProvider: ({ children }: { children: React.ReactNode }) => children,
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() }),
}))

import { consciousnessController } from '@/lib/consciousness-controller'

const mockController = vi.mocked(consciousnessController)

const TestWrapper = ({ children }: { children: React.ReactNode }) => (
  <LocaleProvider>{children}</LocaleProvider>
)

const mockEpisodes = [
  {
    input: 'Test input 1',
    response: 'Response 1',
    narrative: 'Narrative 1',
    qualia: { valence: 0.5, arousal: 0.3, novelty: 0.6, coherence: 0.7, salience: 0.4, certainty: 0.8, complexity: 0.3 },
    growth_delta: 0.05,
    rating: 4,
    timestamp: '2025-01-15T10:00:00Z',
  },
  {
    input: 'Test input 2',
    response: 'Response 2',
    narrative: 'Narrative 2',
    qualia: { valence: 0.4, arousal: 0.2, novelty: 0.5, coherence: 0.6, salience: 0.3, certainty: 0.7, complexity: 0.4 },
    growth_delta: -0.02,
    rating: 2,
    timestamp: '2025-01-15T11:00:00Z',
  },
]

describe('ConsciousnessStatisticsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockController.getEpisodeHistory.mockResolvedValue({ episodes: mockEpisodes } as any)
    mockController.getStatus.mockResolvedValue({ response_quality: 0.8, last_reflection: 'Test reflection' } as any)
    mockController.healthCheck.mockResolvedValue({ health_score: 75, episodes: 10, avg_growth: 0.03, positive_ratio: 0.6 } as any)
    mockController.evaluate.mockResolvedValue({ overall_score: 75, response_quality: 0.8, narrative_quality: 0.7, belief_consistency: 0.9, qualia_coherence: 0.85, diagnostics: ['Good coherence'] } as any)
  })

  it('renders without crashing', async () => {
    render(<ConsciousnessStatisticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_statistics.page_title').length).toBeGreaterThan(0)
    })
  })

  it('shows statistics data', async () => {
    render(<ConsciousnessStatisticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_statistics.health_score').length).toBeGreaterThan(0)
      expect(screen.getAllByText('75').length).toBeGreaterThan(0)
    })
  })

  it('shows export button', async () => {
    render(<ConsciousnessStatisticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_statistics.export_json').length).toBeGreaterThan(0)
    })
  })

  it('displays performance metrics', async () => {
    render(<ConsciousnessStatisticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_statistics.episodes_per_day').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_statistics.avg_rating').length).toBeGreaterThan(0)
    })
  })

  it('displays rating distribution', async () => {
    render(<ConsciousnessStatisticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_statistics.rating_dist_title').length).toBeGreaterThan(0)
    })
  })

  it('displays growth trend chart', async () => {
    render(<ConsciousnessStatisticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_statistics.growth_trend').length).toBeGreaterThan(0)
    })
  })

  it('displays insights section', async () => {
    render(<ConsciousnessStatisticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_statistics.insights_title').length).toBeGreaterThan(0)
    })
  })
})
