import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessPlaygroundPage from './page'
import { consciousnessController } from '@/lib/consciousness-controller'
import type { ConsciousnessController } from '@/lib/consciousness-controller'

const { mockAddToast } = vi.hoisted(() => ({
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    healthCheck: vi.fn(),
    getEpisodeHistory: vi.fn(),
    getStatus: vi.fn(),
    getSelfModel: vi.fn(),
    getQualia: vi.fn(),
    reflect: vi.fn(),
    updateConfig: vi.fn(),
    getTrainingStatus: vi.fn(),
    startTraining: vi.fn(),
    evaluate: vi.fn(),
    getQualiaHistory: vi.fn(),
    getBeliefsHistory: vi.fn(),
    submitFeedback: vi.fn(),
    seedData: vi.fn(),
    getPersonality: vi.fn(),
    updatePersonality: vi.fn(),
    resetPersonality: vi.fn(),
    clearEpisodes: vi.fn(),
    resetBeliefs: vi.fn(),
    getPersonalityHistory: vi.fn(),
    getPersonalityPresets: vi.fn(),
    applyPersonalityPreset: vi.fn(),
    getPersonalityConflicts: vi.fn(),
    listPersonas: vi.fn(),
    savePersona: vi.fn(),
    getPersona: vi.fn(),
    activatePersona: vi.fn(),
    deletePersona: vi.fn(),
    backup: vi.fn(),
    restore: vi.fn(),
    downloadBackup: vi.fn(),
    importBackup: vi.fn(),
    getStats: vi.fn(),
    batch: vi.fn(),
    connectStream: vi.fn(),
  },
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k, locale: 'en', setLocale: vi.fn() }),
  LocaleProvider: ({ children }: { children: React.ReactNode }) => children,
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: Object.assign(
    vi.fn((sel: any) => sel({ addToast: mockAddToast })),
    { getState: () => ({ addToast: mockAddToast }) },
  ),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

function renderPage() {
  return render(
    <LocaleProvider>
      <ConsciousnessPlaygroundPage />
    </LocaleProvider>
  )
}

describe('ConsciousnessPlaygroundPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockAddToast.mockReset()
    vi.mocked(consciousnessController.getStatus).mockResolvedValue({
      enabled: true,
      level: 2,
      episodes: 42,
      current_qualia: { valence: 0.5, arousal: 0.3, novelty: 0.6, coherence: 0.7, salience: 0.4, certainty: 0.3, complexity: 0.5 },
      beliefs: { competence: 0.75, helpfulness: 0.82, creativity: 0.68, accuracy: 0.90, empathy: 0.73 },
      narrative: 'Test narrative',
      response_quality: { avg_growth: 0.045, positive_ratio: 0.82, total: 42, last_growth: 0.032 },
      last_reflection: 'Test reflection',
    } as any)
    vi.mocked(consciousnessController.getEpisodeHistory).mockResolvedValue({
      episodes: [
        { timestamp: Date.now() / 1000 - 100, input_text: 'Hello', response: 'Hi', narrative: 'Test', self_insight: 'Insight', growth_delta: 0.05, qualia: { valence: 0.5 }, rating: 4, index: 0 },
      ],
      total: 1,
    } as any)
  })

  it('renders page title', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('Consciousness Playground').length).toBeGreaterThan(0)
    })
  })

  it('shows input card', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_playground.input_title').length).toBeGreaterThan(0)
    })
  })

  it('shows textarea for input', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_playground.input_title').length).toBeGreaterThan(0)
    })
    const textarea = screen.getAllByRole('textbox')[0]
    expect(textarea).toBeTruthy()
  })

  it('shows preset buttons', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_playground.preset_curious').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_playground.preset_technical').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_playground.preset_creative').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_playground.preset_emotional').length).toBeGreaterThan(0)
    })
  })

  it('shows reflect button', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_playground.reflect').length).toBeGreaterThan(0)
    })
  })

  it('shows process button', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_playground.process').length).toBeGreaterThan(0)
    })
  })

  it('shows qualia dimensions', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_playground.qualia_title').length).toBeGreaterThan(0)
    })
    expect(screen.getByText('Valence')).toBeTruthy()
    expect(screen.getByText('Arousal')).toBeTruthy()
    expect(screen.getByText('Novelty')).toBeTruthy()
  })

  it('shows belief dimensions', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_playground.beliefs_title').length).toBeGreaterThan(0)
    })
    expect(screen.getByText('Competence')).toBeTruthy()
    expect(screen.getByText('Helpfulness')).toBeTruthy()
    expect(screen.getByText('Creativity')).toBeTruthy()
  })

  it('shows episodes card', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_playground.episodes_title').length).toBeGreaterThan(0)
    })
  })

  it('shows status card', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_playground.status_title').length).toBeGreaterThan(0)
    })
    expect(screen.getAllByText('consciousness_playground.episodes_count').length).toBeGreaterThan(0)
    expect(screen.getAllByText('consciousness_playground.avg_growth').length).toBeGreaterThan(0)
    expect(screen.getAllByText('consciousness_playground.positive_ratio').length).toBeGreaterThan(0)
    expect(screen.getAllByText('consciousness_playground.enabled').length).toBeGreaterThan(0)
  })

  it('calls getStatus and getEpisodeHistory on mount', async () => {
    renderPage()
    await waitFor(() => {
      expect(consciousnessController.getStatus).toHaveBeenCalled()
      expect(consciousnessController.getEpisodeHistory).toHaveBeenCalledWith(20)
    })
  })

  it('can type in textarea', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_playground.input_title').length).toBeGreaterThan(0)
    })
    const textarea = screen.getAllByRole('textbox')[0] as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: 'Test input' } })
    expect(textarea.value).toBe('Test input')
  })
})
