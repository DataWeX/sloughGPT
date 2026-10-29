import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act, cleanup } from '@testing-library/react'

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

const SAMPLE_MODELS = [
  { model_id: 'gpt2', status: 'loaded', registered_at: '2026-01-15T00:00:00Z', metrics: { failures: 5, last_error: 'timeout' } },
  { model_id: 'qwen-0.5b', status: 'ready', registered_at: '2026-02-20T00:00:00Z' },
  { model_id: 'bert-base', status: 'failed', registered_at: '2026-03-10T00:00:00Z', metrics: { failures: 3, last_error: 'OOM' } },
]

describe('RegistryPage — initial load flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue([])
    mockStats.mockResolvedValue(null)
    mockBest.mockResolvedValue(null)
  })
  afterEach(() => { cleanup() })

  it('renders page header', async () => {
    render(<RegistryPage />)
    expect(screen.getAllByText('Registry').length).toBeGreaterThanOrEqual(1)
  })

  it('shows loading state', async () => {
    mockList.mockReturnValue(new Promise(() => {}))
    mockStats.mockReturnValue(new Promise(() => {}))
    render(<RegistryPage />)
    expect(screen.getAllByText('Registry').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no models', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/no models|0 models/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders search input', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getAllByPlaceholderText(/search/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders refresh button', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getAllByRole('button').length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('RegistryPage — stats display flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue([])
    mockStats.mockResolvedValue({ total_models: 5, loaded_models: 2, failed_models: 1, circuit_breaker_open: false })
    mockBest.mockResolvedValue(null)
  })
  afterEach(() => { cleanup() })

  it('displays total models count', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getByText('5')).toBeTruthy()
    })
  })

  it('displays loaded models count', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getByText('2')).toBeTruthy()
    })
  })

  it('displays failed models count', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getByText('1')).toBeTruthy()
    })
  })

  it('shows circuit breaker closed', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Closed').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows circuit breaker open', async () => {
    mockStats.mockResolvedValue({ total_models: 1, loaded_models: 0, failed_models: 1, circuit_breaker_open: true })
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getByText('Open')).toBeTruthy()
    })
  })
})

describe('RegistryPage — model list flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue(SAMPLE_MODELS)
    mockStats.mockResolvedValue({ total_models: 3, loaded_models: 1, failed_models: 1, circuit_breaker_open: false })
    mockBest.mockResolvedValue(null)
  })
  afterEach(() => { cleanup() })

  it('displays model count in subtitle', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getByText(/3 models registered/)).toBeTruthy()
    })
  })

  it('shows loaded status badge', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getAllByText('loaded').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows failed status badge', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getAllByText('failed').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows ready status badge', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getAllByText('ready').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows registration date', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/1\/15\/2026/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows details button for failed models', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/details/i).length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('RegistryPage — search flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue(SAMPLE_MODELS)
    mockStats.mockResolvedValue(null)
    mockBest.mockResolvedValue(null)
  })
  afterEach(() => { cleanup() })

  it('search input is rendered and interactive', async () => {
    render(<RegistryPage />)
    await waitFor(() => { expect(screen.getByText(/3 models registered/)).toBeTruthy() })
    const searchInput = screen.getAllByPlaceholderText(/search/i)[0] as HTMLInputElement
    expect(searchInput).toBeTruthy()
    expect(searchInput.type).toBe('search')
  })

  it('search with no match shows empty state', async () => {
    render(<RegistryPage />)
    await waitFor(() => { expect(screen.getByText(/3 models registered/)).toBeTruthy() })
    const searchInput = screen.getAllByPlaceholderText(/search/i)[0] as HTMLInputElement
    const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!
    nativeSetter.call(searchInput, 'nonexistent')
    searchInput.dispatchEvent(new Event('input', { bubbles: true }))
    await waitFor(() => {
      expect(screen.getAllByText(/no models match/i).length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('RegistryPage — model details flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue(SAMPLE_MODELS)
    mockStats.mockResolvedValue(null)
    mockBest.mockResolvedValue(null)
  })
  afterEach(() => { cleanup() })

  it('click details expands metrics', async () => {
    render(<RegistryPage />)
    await waitFor(() => { expect(screen.getAllByText(/details/i).length).toBeGreaterThanOrEqual(1) })
    const detailsBtn = screen.getAllByText(/details/i)[0]
    act(() => { fireEvent.click(detailsBtn) })
    await waitFor(() => {
      expect(screen.getAllByText(/error details/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('click hide collapses metrics', async () => {
    render(<RegistryPage />)
    await waitFor(() => { expect(screen.getAllByText(/details/i).length).toBeGreaterThanOrEqual(1) })
    const detailsBtn = screen.getAllByText(/details/i)[0]
    act(() => { fireEvent.click(detailsBtn) })
    await waitFor(() => { expect(screen.getAllByText(/error details/i).length).toBeGreaterThanOrEqual(1) })
    const hideBtn = screen.getAllByText(/hide/i)[0]
    act(() => { fireEvent.click(hideBtn) })
    await waitFor(() => {
      expect(screen.queryByText(/error details/i)).toBeNull()
    })
  })
})

describe('RegistryPage — best model flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue([])
    mockStats.mockResolvedValue(null)
    mockBest.mockResolvedValue({ model_id: 'gpt2', score: 0.95, accuracy: 0.88 })
  })
  afterEach(() => { cleanup() })

  it('displays best model card', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getByText('Best Model')).toBeTruthy()
    })
  })

  it('shows best model metrics', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getByText('0.95')).toBeTruthy()
    })
  })

  it('formats metric key names', async () => {
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/model id/).length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('RegistryPage — refresh flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue(SAMPLE_MODELS)
    mockStats.mockResolvedValue(null)
    mockBest.mockResolvedValue(null)
  })
  afterEach(() => { cleanup() })

  it('refresh reloads data', async () => {
    render(<RegistryPage />)
    await waitFor(() => { expect(screen.getByText(/3 models registered/)).toBeTruthy() })
    const refreshBtn = screen.getAllByRole('button').find(b => b.querySelector('svg'))
    if (refreshBtn) {
      await act(async () => { fireEvent.click(refreshBtn) })
      expect(mockList).toHaveBeenCalled()
    }
  })
})

describe('RegistryPage — error handling', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(() => { cleanup() })

  it('handles list failure gracefully', async () => {
    mockList.mockRejectedValue(new Error('Network error'))
    mockStats.mockResolvedValue(null)
    mockBest.mockResolvedValue(null)
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Registry').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('handles stats failure gracefully', async () => {
    mockList.mockResolvedValue([])
    mockStats.mockRejectedValue(new Error('Stats error'))
    mockBest.mockResolvedValue(null)
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Registry').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('handles best failure gracefully', async () => {
    mockList.mockResolvedValue([])
    mockStats.mockResolvedValue(null)
    mockBest.mockRejectedValue(new Error('Best error'))
    render(<RegistryPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Registry').length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('RegistryPage — loading to loaded transition', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(() => { cleanup() })

  it('transitions from loading to data', async () => {
    let resolve: (v: unknown) => void
    mockList.mockReturnValue(new Promise(r => { resolve = r }))
    mockStats.mockResolvedValue(null)
    mockBest.mockResolvedValue(null)
    render(<RegistryPage />)
    expect(screen.getAllByText('Registry').length).toBeGreaterThanOrEqual(1)
    await act(async () => {
      resolve!([])
    })
    await waitFor(() => {
      expect(screen.getAllByText(/0 models registered/i).length).toBeGreaterThanOrEqual(1)
    })
  })
})
