/**
 * Tests for the Consciousness History page.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessHistoryPage from './page'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getEpisodeHistory: vi.fn(),
    submitFeedback: vi.fn(),
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

const mockEpisodes = [
  {
    input: 'What is consciousness?',
    response: 'Consciousness is awareness.',
    narrative: 'A philosophical inquiry.',
    qualia: { valence: 0.5, arousal: 0.3 },
    growth_delta: 0.05,
    rating: 4,
    timestamp: '2025-09-11T10:00:00Z',
  },
  {
    input: 'Tell me about AI',
    response: 'AI is artificial intelligence.',
    narrative: 'A technical question.',
    qualia: { valence: 0.6, arousal: 0.4 },
    growth_delta: -0.02,
    rating: 3,
    timestamp: '2025-09-11T11:00:00Z',
  },
]

describe('ConsciousnessHistoryPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockController.getEpisodeHistory.mockResolvedValue({ episodes: mockEpisodes, total: 2 })
    mockController.submitFeedback.mockResolvedValue({ accepted: true, belief_updates: [] })
  })

  it('renders the page', async () => {
    render(<ConsciousnessHistoryPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_history.page_title').length).toBeGreaterThan(0)
    })
  })

  it('displays total episodes stat', async () => {
    render(<ConsciousnessHistoryPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_history.total_episodes').length).toBeGreaterThan(0)
    })
  })

  it('displays filtered count stat', async () => {
    render(<ConsciousnessHistoryPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_history.filtered_count').length).toBeGreaterThan(0)
    })
  })

  it('displays avg rating stat', async () => {
    render(<ConsciousnessHistoryPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_history.avg_rating').length).toBeGreaterThan(0)
    })
  })

  it('displays avg growth stat', async () => {
    render(<ConsciousnessHistoryPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_history.avg_growth').length).toBeGreaterThan(0)
    })
  })

  it('displays rating distribution', async () => {
    render(<ConsciousnessHistoryPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_history.rating_distribution').length).toBeGreaterThan(0)
    })
  })

  it('displays filters section', async () => {
    render(<ConsciousnessHistoryPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_history.filters_title').length).toBeGreaterThan(0)
    })
  })

  it('displays timeline section', async () => {
    render(<ConsciousnessHistoryPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_history.timeline_title').length).toBeGreaterThan(0)
    })
  })

  it('displays episode content', async () => {
    render(<ConsciousnessHistoryPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('What is consciousness?').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Tell me about AI').length).toBeGreaterThan(0)
    })
  })

  it('displays export button', async () => {
    render(<ConsciousnessHistoryPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_history.export').length).toBeGreaterThan(0)
    })
  })

  it('displays import button', async () => {
    render(<ConsciousnessHistoryPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_history.import').length).toBeGreaterThan(0)
    })
  })

  it('displays search input', async () => {
    render(<ConsciousnessHistoryPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByPlaceholderText('consciousness_history.search_placeholder').length).toBeGreaterThan(0)
    })
  })
})
