import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessMasterDashboardPage from './page'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    healthCheck: vi.fn(),
    getStatus: vi.fn(),
    evaluate: vi.fn(),
    getEpisodeHistory: vi.fn(),
    getQualiaHistory: vi.fn(),
    getBeliefsHistory: vi.fn(),
    seedData: vi.fn(),
    reflect: vi.fn(),
  },
}))

vi.mock('@/hooks/useConsciousnessStats', () => ({
  useConsciousnessStats: () => ({
    stats: { total_episodes: 10 },
    loading: false,
  }),
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

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() }),
}))

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  RadarChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  PolarGrid: () => null,
  PolarAngleAxis: () => null,
  PolarRadiusAxis: () => null,
  Radar: () => null,
  BarChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Bar: () => null,
  LineChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Line: () => null,
  AreaChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Area: () => null,
  CartesianGrid: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  Legend: () => null,
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
    rating: 4,
    qualia: { valence: 0.5, arousal: 0.3 },
  },
  {
    timestamp: Date.now() / 1000 - 50,
    input_text: 'Test input 2',
    self_insight: 'Insight 2',
    growth_delta: -0.02,
    rating: 2,
    qualia: { valence: 0.4, arousal: 0.2 },
  },
]

describe('ConsciousnessMasterDashboardPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockController.healthCheck.mockResolvedValue({ score: 85, health_score: 85 } as any)
    mockController.getStatus.mockResolvedValue({ level: 2, enabled: true, episode_count: 10, beliefs: { competence: 0.7, helpfulness: 0.8 } } as any)
    mockController.evaluate.mockResolvedValue({ overall_score: 75, diagnostics: ['Good coherence'] } as any)
    mockController.getEpisodeHistory.mockResolvedValue({ episodes: mockEpisodes } as any)
    mockController.getQualiaHistory.mockResolvedValue({ history: [] } as any)
    mockController.getBeliefsHistory.mockResolvedValue({ beliefs: [] } as any)
    mockController.seedData.mockResolvedValue({ seeded: true })
    mockController.reflect.mockResolvedValue({ reflection: 'Test reflection' })
  })

  it('renders without crashing', async () => {
    render(<ConsciousnessMasterDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_master.page_title').length).toBeGreaterThan(0)
    })
  })

  it('shows master controls', async () => {
    render(<ConsciousnessMasterDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_master.health_score').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_master.total_episodes').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_master.avg_growth').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_master.avg_rating').length).toBeGreaterThan(0)
    })
  })

  it('shows seed button', async () => {
    render(<ConsciousnessMasterDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_master.seed').length).toBeGreaterThan(0)
    })
  })

  it('shows reflect button', async () => {
    render(<ConsciousnessMasterDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_master.reflect').length).toBeGreaterThan(0)
    })
  })

  it('shows export button', async () => {
    render(<ConsciousnessMasterDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_master.export').length).toBeGreaterThan(0)
    })
  })

  it('shows auto-refresh toggle', async () => {
    render(<ConsciousnessMasterDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_master.auto_refresh').length).toBeGreaterThan(0)
    })
  })

  it('displays diagnostics section', async () => {
    render(<ConsciousnessMasterDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_master.diagnostics_title').length).toBeGreaterThan(0)
    })
  })

  it('displays recent episodes section', async () => {
    render(<ConsciousnessMasterDashboardPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_master.recent_title').length).toBeGreaterThan(0)
    })
  })
})
