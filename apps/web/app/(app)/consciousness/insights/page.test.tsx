import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import ConsciousnessInsightsPage from './page'
import { consciousnessController } from '@/lib/consciousness-controller'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getEpisodeHistory: vi.fn(),
    healthCheck: vi.fn(),
    evaluate: vi.fn(),
    getPersonality: vi.fn(),
  },
}))

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn().mockResolvedValue({}),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: vi.fn(() => ({
    t: (key: string) => key,
    locale: 'en',
  })),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: vi.fn(() => ({
    addToast: vi.fn(),
  })),
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconSparkle: (props: any) => <svg {...props} />,
  IconRefresh: (props: any) => <svg {...props} />,
  IconDownload: (props: any) => <svg {...props} />,
}))

const mockEpisodes = {
  episodes: [
    { growth_delta: 0.02, rating: 4, input_text: 'Hello world' },
    { growth_delta: 0.03, rating: 5, input_text: 'Test input' },
    { growth_delta: 0.01, rating: 3, input_text: 'Another message' },
    { growth_delta: 0.04, rating: 5, input_text: 'Final test' },
  ],
  total: 4,
}

const mockHealth = {
  health_score: 0.85,
  enabled: true,
  level: 2,
  episodes: 42,
  avg_growth: 0.02,
  positive_ratio: 0.75,
  qualia: { valence: 0.5, arousal: 0.3, novelty: 0.6, coherence: 0.7 },
  last_reflection: 'Reflecting on growth',
  diagnostics: [],
}

const mockEval = {
  overall_score: 75,
  metrics: {
    coherence: { score: 80, weight: 0.2, details: 'Good' },
    growth: { score: 70, weight: 0.15, details: 'OK' },
  },
  diagnostics: [],
}

const mockPersonality = {
  name: 'Friendly',
  description: 'A friendly personality',
  traits: { openness: 0.7, agreeableness: 0.8 },
}

describe('ConsciousnessInsightsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(consciousnessController.getEpisodeHistory).mockResolvedValue(mockEpisodes as any)
    vi.mocked(consciousnessController.healthCheck).mockResolvedValue(mockHealth as any)
    vi.mocked(consciousnessController.evaluate).mockResolvedValue(mockEval as any)
    vi.mocked(consciousnessController.getPersonality).mockResolvedValue(mockPersonality as any)
  })

  it('renders without crashing', async () => {
    render(<ConsciousnessInsightsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_insights.page_title').length).toBeGreaterThan(0)
    })
  })

  it('shows insight cards', async () => {
    render(<ConsciousnessInsightsPage />)
    await waitFor(() => {
      const cards = document.querySelectorAll('[class*="cursor-pointer"]')
      expect(cards.length).toBeGreaterThan(0)
    })
  })

  it('shows composite score', async () => {
    render(<ConsciousnessInsightsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_insights.overall_score').length).toBeGreaterThan(0)
      expect(screen.getAllByText('/100').length).toBeGreaterThan(0)
    })
  })

  it('has refresh button', async () => {
    render(<ConsciousnessInsightsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_insights.refresh').length).toBeGreaterThan(0)
    })
  })

  it('has export button', async () => {
    render(<ConsciousnessInsightsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_insights.export').length).toBeGreaterThan(0)
    })
  })
})
