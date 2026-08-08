import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const mockList = vi.fn()
const mockListCheckpoints = vi.fn()
const mockSwitch = vi.fn()
const mockLoadCheckpoint = vi.fn()
const mockGetTraitWeights = vi.fn()
const mockListWeightSnapshots = vi.fn()
const mockSaveWeightSnapshot = vi.fn()
const mockLoadWeightSnapshot = vi.fn()
const mockDeleteWeightSnapshot = vi.fn()

vi.mock('@/lib/souls-controller', () => ({
  soulsController: {
    list: (...args: unknown[]) => mockList(...args),
    listCheckpoints: (...args: unknown[]) => mockListCheckpoints(...args),
    switch: (...args: unknown[]) => mockSwitch(...args),
    loadCheckpoint: (...args: unknown[]) => mockLoadCheckpoint(...args),
    getTraitWeights: (...args: unknown[]) => mockGetTraitWeights(...args),
    listWeightSnapshots: (...args: unknown[]) => mockListWeightSnapshots(...args),
    saveWeightSnapshot: (...args: unknown[]) => mockSaveWeightSnapshot(...args),
    loadWeightSnapshot: (...args: unknown[]) => mockLoadWeightSnapshot(...args),
    deleteWeightSnapshot: (...args: unknown[]) => mockDeleteWeightSnapshot(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

import SoulsPage from './page'

describe('SoulsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue({ souls: [], current_soul: null })
    mockListCheckpoints.mockResolvedValue({ checkpoints: [] })
  })

  it('renders page header', async () => {
    render(<SoulsPage />)
    expect(screen.getAllByText('Souls').length).toBeGreaterThanOrEqual(1)
  })

  it('renders tab buttons', async () => {
    render(<SoulsPage />)
    await screen.findAllByText('Personalities')
    expect(screen.getAllByText('Souls').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Checkpoints').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Weights').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Snapshots').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty souls state', async () => {
    render(<SoulsPage />)
    await screen.findAllByText('No personalities found.')
  })

  it('displays soul list when loaded', async () => {
    mockList.mockResolvedValue({
      souls: [{ name: 'Friendly', description: 'A warm companion', traits: ['warmth', 'humor'] }],
      current_soul: null,
    })
    render(<SoulsPage />)
    await screen.findByText('Friendly')
    expect(screen.getByText('A warm companion')).toBeTruthy()
  })

  it('shows active soul badge', async () => {
    mockList.mockResolvedValue({
      souls: [{ name: 'Friendly', description: '', traits: [] }],
      current_soul: 'Friendly',
    })
    render(<SoulsPage />)
    await screen.findByText('active')
  })
})
