import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import ConsciousnessDebugPage from './page'
import { consciousnessController } from '@/lib/consciousness-controller'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getStatus: vi.fn(),
    getSelfModel: vi.fn(),
    getTrainingStatus: vi.fn(),
    healthCheck: vi.fn(),
    evaluate: vi.fn(),
    seedData: vi.fn(),
  },
}))

vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
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
  IconCode: (props: any) => <svg {...props} />,
  IconPlus: (props: any) => <svg {...props} />,
  IconTrash: (props: any) => <svg {...props} />,
  IconClock: (props: any) => <svg {...props} />,
}))

const mockStatus = {
  enabled: true,
  level: 2,
  episodes: 42,
  beliefs: { competence: 0.8 },
  training: { is_training: false, total_pairs: 15 },
}

const mockSelfModel = {
  self_beliefs: { competence: 0.8, helpfulness: 0.9 },
}

const mockTrainStatus = {
  pairs_collected: 15,
  min_pairs: 20,
  is_training: false,
}

const mockHealth = {
  health_score: 0.85,
  enabled: true,
}

const mockEval = {
  overall_score: 75,
  diagnostics: [],
}

describe('ConsciousnessDebugPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(consciousnessController.getStatus).mockResolvedValue(mockStatus as any)
    vi.mocked(consciousnessController.getSelfModel).mockResolvedValue(mockSelfModel as any)
    vi.mocked(consciousnessController.getTrainingStatus).mockResolvedValue(mockTrainStatus as any)
    vi.mocked(consciousnessController.healthCheck).mockResolvedValue(mockHealth as any)
    vi.mocked(consciousnessController.evaluate).mockResolvedValue(mockEval as any)
  })

  it('renders without crashing', async () => {
    render(<ConsciousnessDebugPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_debug.page_title').length).toBeGreaterThan(0)
    })
  })

  it('shows state inspector', async () => {
    render(<ConsciousnessDebugPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_debug.state_inspector').length).toBeGreaterThan(0)
    })
  })

  it('shows event log', async () => {
    render(<ConsciousnessDebugPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_debug.event_log').length).toBeGreaterThan(0)
    })
  })

  it('shows console panel', async () => {
    render(<ConsciousnessDebugPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_debug.console').length).toBeGreaterThan(0)
    })
  })

  it('shows diagnostics panel', async () => {
    render(<ConsciousnessDebugPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_debug.diagnostics').length).toBeGreaterThan(0)
    })
  })

  it('shows auto-refresh toggle', async () => {
    render(<ConsciousnessDebugPage />)
    await waitFor(() => {
      const switches = document.querySelectorAll('[role="switch"]')
      expect(switches.length).toBeGreaterThan(0)
    })
  })

  it('shows state endpoint entries', async () => {
    render(<ConsciousnessDebugPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_debug.state_status').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_debug.state_self_model').length).toBeGreaterThan(0)
    })
  })

  it('shows actions section', async () => {
    render(<ConsciousnessDebugPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_debug.actions').length).toBeGreaterThan(0)
    })
  })

  it('shows reset and seed buttons', async () => {
    render(<ConsciousnessDebugPage />)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_debug.reset_defaults').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_debug.seed_data').length).toBeGreaterThan(0)
    })
  })
})
