import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const mockList = vi.fn()
const mockStats = vi.fn()
const mockBest = vi.fn()

vi.mock('@/lib/registry-controller', () => ({
  registryController: {
    list: (...args: unknown[]) => mockList(...args),
    stats: (...args: unknown[]) => mockStats(...args),
    best: (...args: unknown[]) => mockBest(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

import RegistryPage from './page'

describe('RegistryPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue([])
    mockStats.mockResolvedValue(null)
    mockBest.mockResolvedValue(null)
  })

  it('renders page header', async () => {
    render(<RegistryPage />)
    expect(screen.getAllByText('Registry').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no models', async () => {
    render(<RegistryPage />)
    await screen.findAllByText(/no models|0 models/i)
  })

  it('displays stats when available', async () => {
    mockStats.mockResolvedValue({ total_models: 3, loaded_models: 1, failed_models: 0, circuit_breaker_open: false })
    render(<RegistryPage />)
    await screen.findByText('3')
    expect(screen.getByText('1')).toBeTruthy()
  })

  it('displays registered models', async () => {
    mockList.mockResolvedValue([
      { model_id: 'gpt2', status: 'loaded', registered_at: '2026-01-01' },
    ])
    render(<RegistryPage />)
    await screen.findAllByText('gpt2')
    expect(screen.getAllByText('loaded').length).toBeGreaterThanOrEqual(1)
  })

  it('shows circuit breaker open state', async () => {
    mockStats.mockResolvedValue({ total_models: 1, loaded_models: 0, failed_models: 1, circuit_breaker_open: true })
    render(<RegistryPage />)
    await screen.findByText('Open')
  })
})
