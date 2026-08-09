import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

const mockRun = vi.fn()
const mockQuality = vi.fn()
const mockStats = vi.fn()
const mockHistory = vi.fn()

vi.mock('@/lib/benchmark-controller', () => ({
  benchmarkController: {
    run: (...args: unknown[]) => mockRun(...args),
    quality: (...args: unknown[]) => mockQuality(...args),
    stats: (...args: unknown[]) => mockStats(...args),
    history: (...args: unknown[]) => mockHistory(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

import BenchmarkPage from './page'

describe('BenchmarkPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockRun.mockResolvedValue(null)
    mockQuality.mockResolvedValue(null)
    mockStats.mockResolvedValue(null)
    mockHistory.mockResolvedValue([])
  })

  it('renders page header', async () => {
    render(<BenchmarkPage />)
    expect(screen.getAllByText('Benchmark').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty metrics state', async () => {
    render(<BenchmarkPage />)
    await screen.findAllByText(/no metrics available/i)
  })

  it('shows empty responses state', async () => {
    render(<BenchmarkPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Responses' })[0])
    await screen.findAllByText(/no responses logged yet/i)
  })
})
