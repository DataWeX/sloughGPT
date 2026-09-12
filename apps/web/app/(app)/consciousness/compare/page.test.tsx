import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessComparePage from './page'
import { consciousnessController } from '@/lib/consciousness-controller'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    healthCheck: vi.fn(),
    evaluate: vi.fn(),
    getEpisodeHistory: vi.fn(),
    getPersonalityPresets: vi.fn(),
    applyPersonalityPreset: vi.fn(),
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
    vi.fn((sel: any) => sel({ addToast: vi.fn() })),
    { getState: () => ({ addToast: vi.fn() }) },
  ),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@sloughgpt/strui', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@sloughgpt/strui')>()
  return {
    ...actual,
    Skeleton: ({ className, ...props }: any) => <div className={className} data-testid="skeleton" {...props} />,
  }
})

const mockController = vi.mocked(consciousnessController)

const mockHealthData = {
  health_score: 85,
  enabled: true,
  level: 2,
  episodes: 30,
  avg_growth: 0.05,
  positive_ratio: 0.75,
  qualia: { valence: 0.8, arousal: 0.5, novelty: 0.6, coherence: 0.7, salience: 0.65, certainty: 0.6, complexity: 0.55 },
  beliefs: { 'Self-awareness': 0.7, 'Emotional range': 0.5, 'Context awareness': 0.8 },
}

const mockEvalData = {
  overall_score: 75,
  diagnostics: ['Good coherence', 'Moderate growth'],
  metrics: {
    belief_stability: { score: 80, weight: 0.2, details: 'Stable' },
    narrative_coherence: { score: 70, weight: 0.15, details: 'Coherent' },
  },
}

const mockEpisodesData = {
  episodes: [
    { timestamp: Date.now() / 1000 - 100, growth_delta: 0.05, rating: 4 },
    { timestamp: Date.now() / 1000 - 50, growth_delta: 0.03, rating: 5 },
  ],
  total: 2,
}

const mockPresets = {
  presets: [
    { id: 'default', name: 'Default', description: 'Default personality' },
    { id: 'formal', name: 'Formal', description: 'Professional tone' },
    { id: 'creative', name: 'Creative', description: 'Imaginative and expressive' },
  ],
}

function renderPage() {
  return render(
    <LocaleProvider>
      <ConsciousnessComparePage />
    </LocaleProvider>,
  )
}

describe('ConsciousnessComparePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockController.healthCheck.mockResolvedValue(mockHealthData as any)
    mockController.evaluate.mockResolvedValue(mockEvalData as any)
    mockController.getEpisodeHistory.mockResolvedValue(mockEpisodesData as any)
    mockController.getPersonalityPresets.mockResolvedValue(mockPresets as any)
    mockController.applyPersonalityPreset.mockResolvedValue({ applied: true } as any)
  })

  it('renders loading skeletons initially', () => {
    mockController.healthCheck.mockReturnValue(new Promise(() => {}))
    mockController.evaluate.mockReturnValue(new Promise(() => {}))
    mockController.getEpisodeHistory.mockReturnValue(new Promise(() => {}))
    mockController.getPersonalityPresets.mockReturnValue(new Promise(() => {}))

    const { container } = renderPage()
    const skeletons = container.querySelectorAll('[data-testid="skeleton"]')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('renders page title', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_compare.page_title').length).toBeGreaterThan(0)
    })
  })

  it('shows current config selection', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_compare.config_current').length).toBeGreaterThan(0)
    })
  })

  it('shows default config selection', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_compare.config_default').length).toBeGreaterThan(0)
    })
  })

  it('shows swap button', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_compare.swap').length).toBeGreaterThan(0)
    })
  })

  it('shows apply button', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_compare.apply').length).toBeGreaterThan(0)
    })
  })

  it('calls all fetch functions on mount', async () => {
    renderPage()
    await waitFor(() => {
      expect(mockController.healthCheck).toHaveBeenCalledTimes(1)
      expect(mockController.evaluate).toHaveBeenCalledTimes(1)
      expect(mockController.getEpisodeHistory).toHaveBeenCalledWith(100)
      expect(mockController.getPersonalityPresets).toHaveBeenCalledTimes(1)
    })
  })

  it('swap button swaps configs', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_compare.page_title').length).toBeGreaterThan(0)
    })

    const selects = screen.getAllByRole('combobox')
    expect(selects[0]).toHaveValue('current')
    expect(selects[1]).toHaveValue('default')

    const swapBtn = screen.getAllByText('consciousness_compare.swap')[0]
    fireEvent.click(swapBtn)

    expect(selects[0]).toHaveValue('default')
    expect(selects[1]).toHaveValue('current')
  })

  it('apply button calls applyPersonalityPreset', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_compare.page_title').length).toBeGreaterThan(0)
    })

    const applyBtn = screen.getAllByText('consciousness_compare.apply')[0]
    fireEvent.click(applyBtn)

    await waitFor(() => {
      expect(mockController.applyPersonalityPreset).toHaveBeenCalledWith('default')
    })
  })

  it('shows health scores for both configs', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_compare.page_title').length).toBeGreaterThan(0)
    })

    expect(screen.getAllByText('85').length).toBeGreaterThanOrEqual(1)

    const scoreTexts = screen.getAllByText(/^\d+$/)
    expect(scoreTexts.length).toBeGreaterThanOrEqual(2)
  })

  it('shows qualia dimensions', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_compare.page_title').length).toBeGreaterThan(0)
    })

    expect(screen.getAllByText(/valence/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/arousal/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/novelty/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/coherence/i).length).toBeGreaterThan(0)
  })

  it('shows belief comparisons', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_compare.page_title').length).toBeGreaterThan(0)
    })

    expect(screen.getAllByText('Self-awareness').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Emotional range').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Context awareness').length).toBeGreaterThan(0)
  })
})
