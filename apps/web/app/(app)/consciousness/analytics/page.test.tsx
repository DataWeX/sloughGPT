import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessAnalyticsPage from './page'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getEpisodeHistory: vi.fn(),
    getQualiaHistory: vi.fn(),
    getBeliefsHistory: vi.fn(),
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

const mockQualia = [
  { valence: 0.5, arousal: 0.3, novelty: 0.6, coherence: 0.7, salience: 0.4, certainty: 0.8, complexity: 0.3, timestamp: '2025-01-15T10:00:00Z' },
  { valence: 0.4, arousal: 0.2, novelty: 0.5, coherence: 0.6, salience: 0.3, certainty: 0.7, complexity: 0.4, timestamp: '2025-01-15T11:00:00Z' },
]

const mockBeliefs = {
  labels: ['Step 1', 'Step 2'],
  competence: [0.7, 0.8],
  helpfulness: [0.6, 0.7],
  creativity: [0.5, 0.6],
  accuracy: [0.8, 0.9],
  empathy: [0.4, 0.5],
}

describe('ConsciousnessAnalyticsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockController.getEpisodeHistory.mockResolvedValue({ episodes: mockEpisodes } as any)
    mockController.getQualiaHistory.mockResolvedValue({ qualia: mockQualia } as any)
    mockController.getBeliefsHistory.mockResolvedValue({ beliefs: mockBeliefs } as any)
  })

  it('renders without crashing', async () => {
    render(<ConsciousnessAnalyticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_analytics.page_title').length).toBeGreaterThan(0)
    })
  })

  it('shows analytics data', async () => {
    render(<ConsciousnessAnalyticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_analytics.total_episodes').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_analytics.avg_growth').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_analytics.avg_rating').length).toBeGreaterThan(0)
    })
  })

  it('displays growth chart section', async () => {
    render(<ConsciousnessAnalyticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_analytics.growth_title').length).toBeGreaterThan(0)
    })
  })

  it('displays qualia chart section', async () => {
    render(<ConsciousnessAnalyticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_analytics.qualia_title').length).toBeGreaterThan(0)
    })
  })

  it('displays rating distribution section', async () => {
    render(<ConsciousnessAnalyticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_analytics.rating_title').length).toBeGreaterThan(0)
    })
  })

  it('displays beliefs evolution section', async () => {
    render(<ConsciousnessAnalyticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_analytics.beliefs_title').length).toBeGreaterThan(0)
    })
  })

  it('shows positive ratio', async () => {
    render(<ConsciousnessAnalyticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_analytics.positive_ratio').length).toBeGreaterThan(0)
    })
  })

  it('shows trend card', async () => {
    render(<ConsciousnessAnalyticsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_analytics.trend').length).toBeGreaterThan(0)
    })
  })
})
