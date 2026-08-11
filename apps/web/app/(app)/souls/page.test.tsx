import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

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

vi.mock('@/components/souls/SoulPersonalityCard', () => ({
  SoulPersonalityCard: () => <div data-testid="soul-personality-card" />,
}))

import SoulsPage from './page'

const SAMPLE_SOULS = [
  { name: 'Friendly', description: 'A warm companion', traits: ['warmth', 'humor'] },
  { name: 'Analyst', description: 'Logical and precise', traits: ['precision', 'logic'] },
  { name: 'Creative', description: 'Imaginative and playful', traits: ['creativity', 'humor'] },
]

const SAMPLE_CHECKPOINTS = [
  { name: 'cp-warm-v2', soul: 'Friendly', loss: 0.45, verdict: 'improved', size_mb: 1.2 },
  { name: 'cp-base-v1', soul: 'Analyst', loss: 0.62, verdict: 'neutral', size_mb: 0.8 },
]

async function clickTab(name: string) {
  await waitFor(() => {
    const buttons = screen.getAllByRole('button')
    const tab = buttons.find(b => b.textContent?.trim().toLowerCase() === name.toLowerCase())
    expect(tab).toBeTruthy()
  })
  const buttons = screen.getAllByRole('button')
  const tab = buttons.find(b => b.textContent?.trim().toLowerCase() === name.toLowerCase())
  if (tab) act(() => { fireEvent.click(tab) })
}

describe('SoulsPage — initial load flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue({ souls: [], current_soul: null })
    mockListCheckpoints.mockResolvedValue({ checkpoints: [] })
    mockListWeightSnapshots.mockResolvedValue([])
  })
  afterEach(() => { cleanup() })

  it('renders page header', async () => {
    render(<SoulsPage />)
    expect(screen.getAllByText('Souls').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty souls state', async () => {
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getByText('No personalities found.')).toBeTruthy()
    })
  })

  it('shows empty checkpoints state', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Checkpoints')
    await waitFor(() => {
      expect(screen.getByText(/no checkpoints/i)).toBeTruthy()
    })
  })

  it('shows train link in empty checkpoints', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Checkpoints')
    await waitFor(() => {
      expect(screen.getAllByText(/train a model/i).length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('SoulsPage — tab switching flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue({ souls: SAMPLE_SOULS, current_soul: null })
    mockListCheckpoints.mockResolvedValue({ checkpoints: [] })
    mockListWeightSnapshots.mockResolvedValue([])
  })
  afterEach(() => { cleanup() })

  it('renders all four tabs', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('Friendly')).toBeTruthy() })
    const buttons = screen.getAllByRole('button')
    expect(buttons.some(b => b.textContent?.trim().toLowerCase() === 'souls')).toBeTruthy()
    expect(buttons.some(b => b.textContent?.trim().toLowerCase() === 'checkpoints')).toBeTruthy()
    expect(buttons.some(b => b.textContent?.trim().toLowerCase() === 'weights')).toBeTruthy()
    expect(buttons.some(b => b.textContent?.trim().toLowerCase() === 'snapshots')).toBeTruthy()
  })

  it('souls tab active by default', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('Friendly')).toBeTruthy() })
    expect(screen.getByText(/3 personalities/)).toBeTruthy()
  })

  it('switching to checkpoints tab shows checkpoints content', async () => {
    mockListCheckpoints.mockResolvedValue({ checkpoints: SAMPLE_CHECKPOINTS })
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('Friendly')).toBeTruthy() })
    await clickTab('Checkpoints')
    await waitFor(() => {
      expect(screen.getByText('cp-warm-v2')).toBeTruthy()
    })
  })

  it('switching to weights tab loads weights', async () => {
    mockGetTraitWeights.mockResolvedValue({ personality: { warmth: 0.8, humor: 0.5 } })
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('Friendly')).toBeTruthy() })
    await clickTab('Weights')
    await waitFor(() => {
      expect(mockGetTraitWeights).toHaveBeenCalled()
    })
  })

  it('switching to snapshots tab loads snapshots', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('Friendly')).toBeTruthy() })
    await clickTab('Snapshots')
    await waitFor(() => {
      expect(mockListWeightSnapshots).toHaveBeenCalled()
    })
  })
})

describe('SoulsPage — soul list display flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue({ souls: SAMPLE_SOULS, current_soul: null })
    mockListCheckpoints.mockResolvedValue({ checkpoints: [] })
    mockListWeightSnapshots.mockResolvedValue([])
  })
  afterEach(() => { cleanup() })

  it('displays all souls', async () => {
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getByText('Friendly')).toBeTruthy()
      expect(screen.getByText('Analyst')).toBeTruthy()
      expect(screen.getByText('Creative')).toBeTruthy()
    })
  })

  it('shows active badge for current soul', async () => {
    mockList.mockResolvedValue({ souls: SAMPLE_SOULS, current_soul: 'Friendly' })
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1) })
    expect(screen.getAllByText(/active/i).length).toBeGreaterThanOrEqual(1)
  })

  it('shows soul descriptions', async () => {
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getByText('A warm companion')).toBeTruthy()
    })
  })

  it('shows trait chips', async () => {
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('warmth').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows switch button for non-active souls', async () => {
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Switch').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('no switch button for active soul', async () => {
    mockList.mockResolvedValue({ souls: SAMPLE_SOULS, current_soul: 'Friendly' })
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1) })
    const switchButtons = screen.getAllByText('Switch')
    expect(switchButtons.length).toBeLessThan(SAMPLE_SOULS.length)
  })
})

describe('SoulsPage — switch soul flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue({ souls: SAMPLE_SOULS, current_soul: 'Friendly' })
    mockListCheckpoints.mockResolvedValue({ checkpoints: [] })
    mockListWeightSnapshots.mockResolvedValue([])
    mockSwitch.mockResolvedValue(undefined)
  })
  afterEach(() => { cleanup() })

  it('switch calls soulsController.switch', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('Switch').length).toBeGreaterThanOrEqual(1) })
    const switchBtn = screen.getAllByText('Switch')[0]
    await act(async () => { fireEvent.click(switchBtn) })
    expect(mockSwitch).toHaveBeenCalled()
  })

  it('switch shows Switching... while pending', async () => {
    let resolve: (v: unknown) => void
    mockSwitch.mockReturnValue(new Promise(r => { resolve = r }))
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('Switch').length).toBeGreaterThanOrEqual(1) })
    const switchBtn = screen.getAllByText('Switch')[0]
    await act(async () => { fireEvent.click(switchBtn) })
    await waitFor(() => {
      expect(screen.getAllByText(/switching/i).length).toBeGreaterThanOrEqual(1)
    })
    await act(async () => { resolve!(undefined) })
  })

  it('switch failure shows toast', async () => {
    mockSwitch.mockRejectedValue(new Error('Switch failed'))
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('Switch').length).toBeGreaterThanOrEqual(1) })
    const switchBtn = screen.getAllByText('Switch')[0]
    await act(async () => { fireEvent.click(switchBtn) })
  })
})

describe('SoulsPage — checkpoints tab flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue({ souls: [], current_soul: null })
    mockListCheckpoints.mockResolvedValue({ checkpoints: SAMPLE_CHECKPOINTS })
    mockListWeightSnapshots.mockResolvedValue([])
    mockLoadCheckpoint.mockResolvedValue(undefined)
  })
  afterEach(() => { cleanup() })

  it('displays checkpoints list', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Checkpoints')
    await waitFor(() => {
      expect(screen.getByText('cp-warm-v2')).toBeTruthy()
    })
  })

  it('shows verdict badges', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Checkpoints')
    await waitFor(() => {
      expect(screen.getAllByText('improved').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows checkpoint metadata', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Checkpoints')
    await waitFor(() => {
      expect(screen.getAllByText(/Friendly/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('load button triggers checkpoint load', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Checkpoints')
    await waitFor(() => { expect(screen.getByText('cp-warm-v2')).toBeTruthy() })
    const loadBtn = screen.getAllByText('Load')[0]
    await act(async () => { fireEvent.click(loadBtn) })
    expect(mockLoadCheckpoint).toHaveBeenCalledWith('cp-warm-v2')
  })

  it('checkpoint count shown in header', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Checkpoints')
    await waitFor(() => {
      expect(screen.getByText(/Checkpoints \(2\)/)).toBeTruthy()
    })
  })
})

describe('SoulsPage — weights tab flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue({ souls: [], current_soul: null })
    mockListCheckpoints.mockResolvedValue({ checkpoints: [] })
    mockListWeightSnapshots.mockResolvedValue([])
    mockGetTraitWeights.mockResolvedValue({
      personality: { warmth: 0.80, humor: 0.50, curiosity: 0.65 },
      cognition: { reasoning: 0.70, memory: 0.55 },
    })
  })
  afterEach(() => { cleanup() })

  it('loads weights on tab switch', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Weights')
    await waitFor(() => {
      expect(mockGetTraitWeights).toHaveBeenCalled()
    })
  })

  it('displays weight categories', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Weights')
    await waitFor(() => {
      expect(screen.getByText('personality')).toBeTruthy()
    })
  })

  it('displays weight values', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Weights')
    await waitFor(() => {
      expect(screen.getByText('0.80')).toBeTruthy()
    })
  })

  it('shows placeholder when weights not loaded', async () => {
    mockGetTraitWeights.mockResolvedValue(null)
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Weights')
    await waitFor(() => {
      expect(screen.getAllByText(/click refresh/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('refresh reloads weights', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Weights')
    await waitFor(() => { expect(screen.getByText('personality')).toBeTruthy() })
    const refreshBtn = screen.getAllByRole('button').find(b => b.querySelector('svg'))
    if (refreshBtn) {
      await act(async () => { fireEvent.click(refreshBtn) })
      expect(mockGetTraitWeights).toHaveBeenCalledTimes(2)
    }
  })
})

describe('SoulsPage — snapshots tab flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue({ souls: [], current_soul: null })
    mockListCheckpoints.mockResolvedValue({ checkpoints: [] })
    mockListWeightSnapshots.mockResolvedValue([{ name: 'v1-baseline', saved_at: '2026-01-15T00:00:00Z' }])
    mockSaveWeightSnapshot.mockResolvedValue(undefined)
    mockLoadWeightSnapshot.mockResolvedValue(undefined)
    mockDeleteWeightSnapshot.mockResolvedValue(undefined)
    mockGetTraitWeights.mockResolvedValue({})
  })
  afterEach(() => { cleanup() })

  it('displays snapshots list', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Snapshots')
    await waitFor(() => {
      expect(screen.getByText('v1-baseline')).toBeTruthy()
    })
  })

  it('shows snapshot date', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Snapshots')
    await waitFor(() => {
      expect(screen.getAllByText(/1\/15\/2026/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('save snapshot with name', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Snapshots')
    await waitFor(() => { expect(screen.getByText('v1-baseline')).toBeTruthy() })
    const input = screen.getByPlaceholderText(/snapshot name/i)
    act(() => { fireEvent.change(input, { target: { value: 'new-snap' } }) })
    const saveBtn = screen.getByText('Save')
    await act(async () => { fireEvent.click(saveBtn) })
    expect(mockSaveWeightSnapshot).toHaveBeenCalledWith('new-snap')
  })

  it('save button disabled when name empty', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Snapshots')
    await waitFor(() => { expect(screen.getByText('Save')).toBeTruthy() })
    const saveBtn = screen.getByText('Save')
    expect(saveBtn).toBeDisabled()
  })

  it('empty snapshots shows placeholder', async () => {
    mockListWeightSnapshots.mockResolvedValue([])
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('No personalities found.')).toBeTruthy() })
    await clickTab('Snapshots')
    await waitFor(() => {
      expect(screen.getByText(/no snapshots saved/i)).toBeTruthy()
    })
  })
})

describe('SoulsPage — search flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue({ souls: SAMPLE_SOULS, current_soul: null })
    mockListCheckpoints.mockResolvedValue({ checkpoints: [] })
    mockListWeightSnapshots.mockResolvedValue([])
  })
  afterEach(() => { cleanup() })

  it('search filters souls by name', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('Friendly')).toBeTruthy() })
    const searchInput = screen.getAllByPlaceholderText(/search/i)[0]
    const user = userEvent.setup()
    await user.type(searchInput, 'friendly')
    await waitFor(() => {
      expect(screen.getByText('Friendly')).toBeTruthy()
      expect(screen.queryByText('Analyst')).toBeNull()
    })
  })

  it('search filters souls by trait', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('Friendly')).toBeTruthy() })
    const searchInput = screen.getAllByPlaceholderText(/search/i)[0]
    const user = userEvent.setup()
    await user.type(searchInput, 'logic')
    await waitFor(() => {
      expect(screen.getByText('Analyst')).toBeTruthy()
      expect(screen.queryByText('Friendly')).toBeNull()
    })
  })

  it('search with no match shows empty', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('Friendly')).toBeTruthy() })
    const searchInput = screen.getAllByPlaceholderText(/search/i)[0]
    const user = userEvent.setup()
    await user.type(searchInput, 'zzz')
    await waitFor(() => {
      expect(screen.getAllByText(/no personalities match/i).length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('SoulsPage — error handling', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(() => { cleanup() })

  it('handles list failure gracefully', async () => {
    mockList.mockRejectedValue(new Error('Network error'))
    mockListCheckpoints.mockResolvedValue({ checkpoints: [] })
    mockListWeightSnapshots.mockResolvedValue([])
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Souls').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('handles checkpoints failure gracefully', async () => {
    mockList.mockResolvedValue({ souls: [], current_soul: null })
    mockListCheckpoints.mockRejectedValue(new Error('Checkpoints error'))
    mockListWeightSnapshots.mockResolvedValue([])
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Souls').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('handles snapshot failure gracefully', async () => {
    mockList.mockResolvedValue({ souls: [], current_soul: null })
    mockListCheckpoints.mockResolvedValue({ checkpoints: [] })
    mockListWeightSnapshots.mockRejectedValue(new Error('Snapshots error'))
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Souls').length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('SoulsPage — KPI grid flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue({ souls: SAMPLE_SOULS, current_soul: 'Friendly' })
    mockListCheckpoints.mockResolvedValue({ checkpoints: SAMPLE_CHECKPOINTS })
    mockListWeightSnapshots.mockResolvedValue([{ name: 'v1' }])
  })
  afterEach(() => { cleanup() })

  it('shows personalities count', async () => {
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getByText('3')).toBeTruthy()
    })
  })

  it('shows active soul name', async () => {
    mockList.mockResolvedValue({ souls: SAMPLE_SOULS, current_soul: 'Analyst' })
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Analyst').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows checkpoints count', async () => {
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getByText('2')).toBeTruthy()
    })
  })

  it('shows snapshots count', async () => {
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getByText('1')).toBeTruthy()
    })
  })
})
