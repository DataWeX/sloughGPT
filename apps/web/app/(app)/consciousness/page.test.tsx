/**
 * Tests for the Consciousness settings page.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ConsciousnessPage from './page'

const mockStatus = {
  enabled: true,
  level: 1,
  current_qualia: {
    valence: 0.5,
    arousal: 0.3,
    novelty: 0.6,
    coherence: 0.7,
    salience: 0.4,
    certainty: 0.3,
    complexity: 0.5,
  },
  beliefs: { helpful: 0.9, accurate: 0.7 },
  episodes: 10,
  narrative: 'Test narrative',
  training: {
    is_training: false,
    total_pairs: 15,
    current_epoch: 0,
    loss: 0.0,
  },
}

const mockEval = {
  overall_score: 75,
  metrics: {
    coherence: { score: 80, weight: 0.2, details: 'Good' },
    growth: { score: 70, weight: 0.15, details: 'OK' },
  },
}

const mockGetStatus = vi.fn()
const mockReflect = vi.fn()
const mockEvaluate = vi.fn()
const mockGetEpisodeHistory = vi.fn()
const mockSeedData = vi.fn()

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getStatus: (...a: unknown[]) => mockGetStatus(...a),
    reflect: (...a: unknown[]) => mockReflect(...a),
    evaluate: (...a: unknown[]) => mockEvaluate(...a),
    getEpisodeHistory: (...a: unknown[]) => mockGetEpisodeHistory(...a),
    seedData: (...a: unknown[]) => mockSeedData(...a),
    updateConfig: vi.fn().mockResolvedValue({ updated: true }),
    startTraining: vi.fn().mockResolvedValue({ started: true }),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: vi.fn() }),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), back: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => '/consciousness',
}))

describe('ConsciousnessPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGetStatus.mockResolvedValue(mockStatus)
    mockReflect.mockResolvedValue({ reflection: 'I am reflecting.' })
    mockEvaluate.mockResolvedValue(mockEval)
    mockGetEpisodeHistory.mockResolvedValue({ episodes: [] })
    mockSeedData.mockResolvedValue({ seeded: true })
  })

  it('renders the page title', async () => {
    render(<ConsciousnessPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Consciousness Level').length).toBeGreaterThan(0)
    })
  })

  it('displays qualia metrics', async () => {
    render(<ConsciousnessPage />)
    await waitFor(() => {
      expect(screen.getAllByText('valence').length).toBeGreaterThan(0)
      expect(screen.getAllByText('coherence').length).toBeGreaterThan(0)
    })
  })

  it('displays beliefs', async () => {
    render(<ConsciousnessPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Self-Model Beliefs').length).toBeGreaterThan(0)
      expect(screen.getAllByText('helpful').length).toBeGreaterThan(0)
      expect(screen.getAllByText('90%').length).toBeGreaterThan(0)
    })
  })

  it('shows reflect button', async () => {
    render(<ConsciousnessPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Reflect').length).toBeGreaterThan(0)
    })
  })

  it('calls reflect endpoint on click', async () => {
    const user = userEvent.setup()
    render(<ConsciousnessPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Reflect').length).toBeGreaterThan(0)
    })
    const reflectBtns = screen.getAllByText('Reflect')
    await user.click(reflectBtns[0])
    await waitFor(() => {
      expect(screen.getAllByText('I am reflecting.').length).toBeGreaterThan(0)
    })
  })

  it('displays evaluation report', async () => {
    render(<ConsciousnessPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Evaluation Report').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Score: 75/100').length).toBeGreaterThan(0)
    })
  })

  it('shows training status', async () => {
    render(<ConsciousnessPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Training Status').length).toBeGreaterThan(0)
      expect(screen.getAllByText('15').length).toBeGreaterThan(0)
    })
  })
})
