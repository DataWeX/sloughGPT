import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

const mockHealthState = vi.fn()
const mockGetCurrent = vi.fn()
const mockGetTraitWeights = vi.fn()

vi.mock('@/hooks/useLiveStatus', () => ({
  useLiveStatus: () => {
    const h = mockHealthState()
    if (h === 'offline') return { connectionStatus: 'offline', health: null, healthLegacy: 'offline', lastUpdate: null, failureCount: 1, connected: false, live: false }
    if (h === null) return { connectionStatus: 'connecting', health: null, healthLegacy: null, lastUpdate: null, failureCount: 0, connected: false, live: false }
    return { connectionStatus: 'connected', health: h, healthLegacy: h, lastUpdate: Date.now(), failureCount: 0, connected: true, live: true }
  },
  liveStatusStore: { getState: vi.fn(() => ({ connectionStatus: 'connected', health: null, healthLegacy: null, lastUpdate: null, failureCount: 0 })), subscribe: vi.fn(() => vi.fn()) },
}))

vi.mock('@/lib/souls-controller', () => ({
  soulsController: {
    getCurrent: (...a: any[]) => mockGetCurrent(...a),
    getTraitWeights: (...a: any[]) => mockGetTraitWeights(...a),
  },
}))

vi.mock('@/components/souls/PersonalitySummary', () => ({
  deriveArchetype: vi.fn(() => ({ label: 'The Optimist' })),
}))

const mockGetUnseenCount = vi.fn()

vi.mock('@/components/WhatsNewDialog', () => ({
  getUnseenCount: () => mockGetUnseenCount(),
}))

const mockHealthSummary = { score: 85, summary: 'Healthy', tokens_per_sec: 15, model_loaded: true, model_type: 'gpt2', soul: 'friendly', uptime_seconds: 100, request_count: 50, error_count: 0, cpu_percent: 30, memory_percent: 40 }

const { useApiMonitor: _useApiMonitor, setHealthSummaryData } = vi.hoisted(() => {
  let hc: typeof mockHealthSummary | null = null
  return {
    useApiMonitor: (selector: (s: any) => any) => selector({ healthSummary: hc, recentFailures: [], failureCount: 0, lastOffline: null }),
    setHealthSummaryData: (v: typeof mockHealthSummary | null) => { hc = v },
  }
})

vi.mock('@/lib/api-monitor-store', () => ({
  useApiMonitor: _useApiMonitor,
}))

import { StatusBar } from './StatusBar'

describe('StatusBar', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setHealthSummaryData(null)
    mockGetUnseenCount.mockReturnValue(0)
    mockHealthState.mockReturnValue({ model_loaded: true, model_type: 'gpt2', inference_count: 42, health_score: 85, health_status: 'healthy', health_summary: 'Healthy', tokens_per_sec: 15, is_inferencing: false, cpu_percent: 30, memory_percent: 40, uptime_seconds: 100, request_count: 50, error_count: 0, soul: 'friendly' })
    mockGetCurrent.mockResolvedValue({ name: 'friendly', description: 'Friendly soul' })
    mockGetTraitWeights.mockResolvedValue({ personality: { warmth: 0.7 }, cognition: {}, emotion: {} })
  })

  afterEach(cleanup)

  it('renders status text', () => {
    render(<StatusBar />)
    expect(screen.getByText('Healthy')).toBeDefined()
  })

  it('renders archetype badge when soul and weights available', async () => {
    render(<StatusBar />)
    const badge = await screen.findAllByText('The Optimist')
    expect(badge.length).toBeGreaterThanOrEqual(1)
  })

  it('renders tokens per second from live health', async () => {
    mockHealthState.mockReturnValue({ model_loaded: true, model_type: 'gpt2', inference_count: 42, health_score: 85, health_status: 'healthy', health_summary: 'Healthy', tokens_per_sec: 15, is_inferencing: false, cpu_percent: 30, memory_percent: 40, uptime_seconds: 100, request_count: 50, error_count: 0, soul: 'friendly' })
    render(<StatusBar />)
    const tps = await screen.findAllByText('15 t/s')
    expect(tps.length).toBeGreaterThanOrEqual(1)
  })

  it('renders inference count when no tps', async () => {
    mockHealthState.mockReturnValue({ model_loaded: true, model_type: 'gpt2', inference_count: 42, health_score: 85, health_status: 'healthy', health_summary: 'Healthy', tokens_per_sec: 0, is_inferencing: false, cpu_percent: 30, memory_percent: 40, uptime_seconds: 100, request_count: 50, error_count: 0, soul: 'friendly' })
    render(<StatusBar />)
    const count = await screen.findAllByText('42 responses')
    expect(count.length).toBeGreaterThanOrEqual(1)
  })

  it('shows offline status', () => {
    mockHealthState.mockReturnValue('offline')
    render(<StatusBar />)
    expect(screen.getByText(/Offline/)).toBeDefined()
  })

  it('links to monitoring page', () => {
    render(<StatusBar />)
    const link = screen.getByRole('link')
    expect(link.getAttribute('href')).toBe('/monitoring')
  })

  it('dispatches toggle-whatsnew event when whats-new button is clicked', () => {
    const listener = vi.fn()
    window.addEventListener('toggle-whatsnew', listener)
    render(<StatusBar />)
    const btn = screen.getByLabelText("What's new")
    fireEvent.click(btn)
    expect(listener).toHaveBeenCalled()
    window.removeEventListener('toggle-whatsnew', listener)
  })

  it('shows unseen count badge when there are new features', () => {
    mockGetUnseenCount.mockReturnValue(3)
    render(<StatusBar />)
    expect(screen.getByText('3')).toBeDefined()
  })

  it('caps unseen count badge at 9+', () => {
    mockGetUnseenCount.mockReturnValue(15)
    render(<StatusBar />)
    expect(screen.getByText('9+')).toBeDefined()
  })
})
