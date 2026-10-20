import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import ConsciousnessPersonalityPage from './page'
import { consciousnessController } from '@/lib/consciousness-controller'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getPersonality: vi.fn(),
    getPersonalityPresets: vi.fn(),
    getPersonalityConflicts: vi.fn(),
    getPersonalityHistory: vi.fn(),
    updatePersonality: vi.fn(),
    resetPersonality: vi.fn(),
    applyPersonalityPreset: vi.fn(),
  },
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

const mockPersonality = {
  values: ['honesty'],
  goals: ['accuracy'],
  voice: { formality: 0.5 },
  style: { use_examples: true },
  interests: ['science'],
  avoid: ['speculation'],
  traits: {
    openness: 0.7,
    conscientiousness: 0.6,
    extraversion: 0.5,
    agreeableness: 0.8,
    neuroticism: 0.3,
    empathy: 0.9,
    humor: 0.4,
    creativity: 0.65,
  },
}

const mockPresets = {
  presets: [
    { name: 'Default', traits: {} },
    { name: 'Formal', traits: {} },
    { name: 'Creative', traits: {} },
  ],
  names: ['Default', 'Formal', 'Creative'],
}

const mockConflicts = {
  conflicts: ['High empathy may reduce decisiveness'],
}

const mockHistory = {
  history: [
    {
      timestamp: 1736935200,
      step: 0,
      voice: { formality: 0.5, warmth: 0.7 },
      traits: { openness: 0.7, agreeableness: 0.8 },
    },
  ],
}

describe('ConsciousnessPersonalityPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(consciousnessController.getPersonality).mockResolvedValue(mockPersonality as any)
    vi.mocked(consciousnessController.getPersonalityPresets).mockResolvedValue(mockPresets as any)
    vi.mocked(consciousnessController.getPersonalityConflicts).mockResolvedValue(mockConflicts as any)
    vi.mocked(consciousnessController.getPersonalityHistory).mockResolvedValue(mockHistory as any)
  })

  it('renders without crashing', async () => {
    render(<ConsciousnessPersonalityPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_personality.page_title').length).toBeGreaterThan(0)
    })
  })

  it('shows personality overview', async () => {
    render(<ConsciousnessPersonalityPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_personality.overview_title').length).toBeGreaterThan(0)
    })
  })

  it('shows trait bars', async () => {
    render(<ConsciousnessPersonalityPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Openness').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Agreeableness').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Empathy').length).toBeGreaterThan(0)
    })
  })

  it('shows presets grid', async () => {
    render(<ConsciousnessPersonalityPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_personality.presets_title').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Default').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Formal').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Creative').length).toBeGreaterThan(0)
    })
  })

  it('has save/reset buttons', async () => {
    render(<ConsciousnessPersonalityPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_personality.reset_defaults').length).toBeGreaterThan(0)
    })
  })
})
