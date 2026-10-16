import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import ConsciousnessDebugPage from './page'
import { consciousnessController } from '@/lib/consciousness-controller'
import { apiGet, apiPost, httpClient } from '@/lib/http-client'

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

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn().mockResolvedValue({}),
  apiPost: vi.fn().mockResolvedValue({}),
  httpClient: { request: vi.fn().mockResolvedValue({}) },
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
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) =>
    selector({ addToast: vi.fn() }),
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

  it('reset posts to the real /consciousness/personality/reset route', async () => {
    render(<ConsciousnessDebugPage />)
    const resetButton = await screen.findByText('consciousness_debug.reset_defaults')
    fireEvent.click(resetButton)
    await waitFor(() => {
      expect(apiPost).toHaveBeenCalledWith(
        '/consciousness/personality/reset',
        undefined,
        expect.anything(),
      )
    })
    // The route that was here before never existed on the backend.
    expect(apiPost).not.toHaveBeenCalledWith('/consciousness/reset', undefined, expect.anything())
  })

  it('console bare text prints a usage hint without any network call', async () => {
    render(<ConsciousnessDebugPage />)
    const input = await screen.findByPlaceholderText('consciousness_debug.console_placeholder')
    fireEvent.change(input, { target: { value: 'process my thoughts' } })
    fireEvent.click(screen.getByText('consciousness_debug.execute'))
    await waitFor(() => {
      expect(screen.getByText(/Unsupported input/)).toBeInTheDocument()
    })
    // No phantom /consciousness/process call, no generic request escape hatch.
    expect(apiPost).not.toHaveBeenCalled()
    expect(httpClient.request).not.toHaveBeenCalled()
  })

  it('console METHOD prefix still routes through the API client', async () => {
    render(<ConsciousnessDebugPage />)
    const input = await screen.findByPlaceholderText('consciousness_debug.console_placeholder')
    fireEvent.change(input, { target: { value: 'GET /consciousness/status' } })
    fireEvent.click(screen.getByText('consciousness_debug.execute'))
    await waitFor(() => {
      expect(httpClient.request).toHaveBeenCalledWith(
        'GET',
        '/consciousness/status',
        undefined,
        expect.anything(),
      )
    })
  })
})
