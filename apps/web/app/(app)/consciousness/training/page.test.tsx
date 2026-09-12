/**
 * Tests for the Consciousness Training page.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessTrainingPage from './page'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getTrainingStatus: vi.fn(),
    evaluate: vi.fn(),
    startTraining: vi.fn(),
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

const mockTrainStatus = {
  pairs_collected: 15,
  min_pairs: 20,
  should_train: false,
  is_training: false,
  training_runs: [
    {
      timestamp: '2025-09-11T10:00:00Z',
      status: 'success',
      pairs_count: 10,
      loss: 0.5,
      adapter_path: '/path/to/adapter',
      elapsed_seconds: 120.5,
    },
  ],
  last_result: {
    status: 'success',
    loss: 0.5,
    adapter_path: '/path/to/adapter',
    elapsed_seconds: 120.5,
  },
}

const mockEvalReport = {
  overall_score: 75,
  narrative_coherence: 80,
  qualia_richness: 70,
  belief_stability: 85,
  self_reflection_depth: 60,
  growth_trajectory: 75,
  curiosity_engagement: 65,
  feedback_alignment: 70,
  episode_count: 50,
  diagnostics: ['Good coherence', 'Moderate growth'],
}

describe('ConsciousnessTrainingPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockController.getTrainingStatus.mockResolvedValue(mockTrainStatus as any)
    mockController.evaluate.mockResolvedValue(mockEvalReport as any)
    mockController.startTraining.mockResolvedValue({ started: true })
  })

  it('renders the page', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_training.page_title').length).toBeGreaterThan(0)
    })
  })

  it('displays training status', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_training.status_title').length).toBeGreaterThan(0)
    })
  })

  it('displays pairs collected', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('15 / 20').length).toBeGreaterThan(0)
    })
  })

  it('displays training runs count', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('1').length).toBeGreaterThan(0)
    })
  })

  it('displays episodes count', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('50').length).toBeGreaterThan(0)
    })
  })

  it('displays last result', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_training.last_result').length).toBeGreaterThan(0)
    })
  })

  it('displays loss value', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('0.5000').length).toBeGreaterThan(0)
    })
  })

  it('displays adapter path', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('/path/to/adapter').length).toBeGreaterThan(0)
    })
  })

  it('displays elapsed time', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('120.5s').length).toBeGreaterThan(0)
    })
  })

  it('displays model path input', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_training.model_path').length).toBeGreaterThan(0)
    })
  })

  it('displays start training button', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_training.start_training').length).toBeGreaterThan(0)
    })
  })

  it('displays evaluation report', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_training.eval_title').length).toBeGreaterThan(0)
    })
  })

  it('displays overall score', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('75').length).toBeGreaterThan(0)
    })
  })

  it('displays eval metrics', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('narrative coherence').length).toBeGreaterThan(0)
      expect(screen.getAllByText('qualia richness').length).toBeGreaterThan(0)
      expect(screen.getAllByText('belief stability').length).toBeGreaterThan(0)
    })
  })

  it('displays diagnostics', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_training.diagnostics').length).toBeGreaterThan(0)
    })
  })

  it('displays training history', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_training.history_title').length).toBeGreaterThan(0)
    })
  })

  it('displays refresh button', async () => {
    render(<ConsciousnessTrainingPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_training.refresh').length).toBeGreaterThan(0)
    })
  })
})
