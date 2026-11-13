import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ConsciousnessMonitorPage from './page'
import { consciousnessController } from '@/lib/consciousness-controller'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getStatus: vi.fn(),
    healthCheck: vi.fn(),
    getEpisodeHistory: vi.fn(),
  },
}))

vi.mock('@/hooks/useConsciousnessLive', () => ({
  useConsciousnessLive: vi.fn(() => ({
    isLive: false,
    lastUpdate: null,
    toggleLive: vi.fn(),
    latestEvent: null,
  })),
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
  IconActivity: (props: any) => <svg {...props} />,
  IconClock: (props: any) => <svg {...props} />,
  IconTrash: (props: any) => <svg {...props} />,
}))

const mockStatus = {
  enabled: true,
  level: 2,
  current_qualia: {
    valence: 0.5,
    arousal: 0.3,
    novelty: 0.6,
    coherence: 0.7,
  },
  beliefs: {
    competence: 0.8,
    helpfulness: 0.9,
    creativity: 0.6,
    accuracy: 0.7,
    empathy: 0.5,
  },
  episodes: 42,
  narrative: 'Test narrative',
  training: {
    is_training: false,
    total_pairs: 15,
    current_epoch: 0,
    loss: 0.0,
  },
}

const mockHealth = {
  health_score: 0.85,
  enabled: true,
  level: 2,
  episodes: 42,
  avg_growth: 0.02,
  positive_ratio: 0.75,
  qualia: { valence: 0.5, arousal: 0.3 },
  last_reflection: 'Reflecting on growth',
  diagnostics: [],
}

const mockEpisodes = {
  episodes: [
    { input: 'Hello', response: 'Hi there', growth_delta: 0.01, rating: 4, timestamp: '2024-01-01T00:00:00Z' },
  ],
  total: 1,
}

describe('ConsciousnessMonitorPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers({ shouldAdvanceTime: true })
    vi.mocked(consciousnessController.getStatus).mockResolvedValue(mockStatus as any)
    vi.mocked(consciousnessController.healthCheck).mockResolvedValue(mockHealth as any)
    vi.mocked(consciousnessController.getEpisodeHistory).mockResolvedValue(mockEpisodes as any)
    Element.prototype.scrollIntoView = vi.fn()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('renders without crashing', async () => {
    render(<ConsciousnessMonitorPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_monitor.page_title').length).toBeGreaterThan(0)
    })
  })

  it('shows connection status indicator', async () => {
    render(<ConsciousnessMonitorPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_monitor.connected').length).toBeGreaterThan(0)
    })
  })

  it('shows qualia bars', async () => {
    render(<ConsciousnessMonitorPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_monitor.qualia').length).toBeGreaterThan(0)
      expect(screen.getAllByText('valence').length).toBeGreaterThan(0)
      expect(screen.getAllByText('arousal').length).toBeGreaterThan(0)
      expect(screen.getAllByText('novelty').length).toBeGreaterThan(0)
      expect(screen.getAllByText('coherence').length).toBeGreaterThan(0)
    })
  })

  it('shows growth stream chart', async () => {
    render(<ConsciousnessMonitorPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_monitor.growth_stream').length).toBeGreaterThan(0)
    })
  })

  it('shows event stream', async () => {
    render(<ConsciousnessMonitorPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_monitor.event_stream').length).toBeGreaterThan(0)
    })
  })

  it('has refresh interval controls', async () => {
    render(<ConsciousnessMonitorPage />)
    await waitFor(() => {
      const rangeInputs = document.querySelectorAll('input[type="range"]')
      expect(rangeInputs.length).toBeGreaterThan(0)
    })
  })

  it('has pause/resume button', async () => {
    render(<ConsciousnessMonitorPage />)
    await waitFor(() => {
      const switches = document.querySelectorAll('[role="switch"]')
      expect(switches.length).toBeGreaterThan(0)
    })
  })

  it('has clear history button', async () => {
    render(<ConsciousnessMonitorPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_monitor.clear_history').length).toBeGreaterThan(0)
    })
  })
})
