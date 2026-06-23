// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const mockHealthState = vi.fn()
const mockGetCurrent = vi.fn()
const mockGetTraitWeights = vi.fn()

vi.mock('@/hooks/useApiHealth', () => ({
  useApiHealth: () => ({ state: mockHealthState() }),
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

import { StatusBar } from './StatusBar'

describe('StatusBar', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockHealthState.mockReturnValue({ model_loaded: true, model_type: 'gpt2', inference_count: 42 })
    mockGetCurrent.mockResolvedValue({ name: 'friendly', description: 'Friendly soul' })
    mockGetTraitWeights.mockResolvedValue({ personality: { warmth: 0.7 }, cognition: {}, emotion: {} })
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ score: 85, summary: 'Healthy', tokens_per_sec: 15 }) })
  })

  afterEach(cleanup)

  it('renders status text', () => {
    render(<StatusBar />)
    expect(screen.getByText('gpt2')).toBeDefined()
  })

  it('renders archetype badge when soul and weights available', async () => {
    render(<StatusBar />)
    const badge = await screen.findAllByText('The Optimist')
    expect(badge.length).toBeGreaterThanOrEqual(1)
  })

  it('renders tokens per second from summary', async () => {
    render(<StatusBar />)
    const tps = await screen.findAllByText('15 t/s')
    expect(tps.length).toBeGreaterThanOrEqual(1)
  })

  it('renders inference count when summary has no tps', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ score: 85, tokens_per_sec: 0 }) })
    render(<StatusBar />)
    const count = await screen.findAllByText('42 responses')
    expect(count.length).toBeGreaterThanOrEqual(1)
  })

  it('shows offline status', () => {
    mockHealthState.mockReturnValue('offline')
    render(<StatusBar />)
    expect(screen.getByText('Offline')).toBeDefined()
  })

  it('links to monitoring page', () => {
    render(<StatusBar />)
    const link = screen.getByRole('link')
    expect(link.getAttribute('href')).toBe('/monitoring')
  })
})
