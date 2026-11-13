/**
 * Tests for the Consciousness Health page.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessHealthPage from './page'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    healthCheck: vi.fn(),
    getStatus: vi.fn(),
    evaluate: vi.fn(),
    getEpisodeHistory: vi.fn(),
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

const mockStatus = {
  response_quality: 75,
  last_reflection: 'Good response quality',
  training: { total_pairs: 20 },
}

const mockEval = {
  overall_score: 75,
  diagnostics: ['Good coherence', 'Moderate growth'],
  metrics: {
    belief_stability: { score: 80, weight: 0.2, details: 'Stable' },
    narrative_coherence: { score: 70, weight: 0.15, details: 'Coherent' },
    feedback_alignment: { score: 65, weight: 0.1, details: 'Aligned' },
  },
}

const mockEpisodes = {
  episodes: [
    { timestamp: Date.now() / 1000 - 100, growth_delta: 0.05 },
    { timestamp: Date.now() / 1000 - 50, growth_delta: 0.03 },
  ],
  total: 2,
}

describe('ConsciousnessHealthPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockController.healthCheck.mockResolvedValue(mockHealth as any)
    mockController.getStatus.mockResolvedValue(mockStatus as any)
    mockController.evaluate.mockResolvedValue(mockEval as any)
    mockController.getEpisodeHistory.mockResolvedValue(mockEpisodes as any)
  })

  it('renders the page', async () => {
    render(<ConsciousnessHealthPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_health.page_title').length).toBeGreaterThan(0)
    })
  })

  it('displays health score', async () => {
    render(<ConsciousnessHealthPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('85').length).toBeGreaterThan(0)
    })
  })

  it('displays enabled status', async () => {
    render(<ConsciousnessHealthPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_health.yes').length).toBeGreaterThan(0)
    })
  })

  it('displays level', async () => {
    render(<ConsciousnessHealthPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('2/3').length).toBeGreaterThan(0)
    })
  })

  it('displays episodes count', async () => {
    render(<ConsciousnessHealthPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('50').length).toBeGreaterThan(0)
    })
  })

  it('displays training pairs', async () => {
    render(<ConsciousnessHealthPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('20').length).toBeGreaterThan(0)
    })
  })

  it('displays avg growth', async () => {
    render(<ConsciousnessHealthPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('+15.0%').length).toBeGreaterThan(0)
    })
  })

  it('displays positive ratio', async () => {
    render(<ConsciousnessHealthPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('80%').length).toBeGreaterThan(0)
    })
  })

  it('displays qualia diversity', async () => {
    render(<ConsciousnessHealthPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('7/7').length).toBeGreaterThan(0)
    })
  })

  it('displays diagnostics section', async () => {
    render(<ConsciousnessHealthPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_health.diagnostics').length).toBeGreaterThan(0)
    })
  })

  it('shows auto-refresh toggle', async () => {
    render(<ConsciousnessHealthPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_health.auto_refresh').length).toBeGreaterThan(0)
    })
  })

  it('displays trend indicator', async () => {
    render(<ConsciousnessHealthPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText(/consciousness_health\.trend/).length).toBeGreaterThan(0)
    })
  })
})
