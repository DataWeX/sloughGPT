import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import DatasetsPage from './page'

const mockList = vi.fn()
const mockSearch = vi.fn()
const mockDelete = vi.fn()
const mockGetStats = vi.fn()

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() }),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/lib/dataset-controller', () => ({
  datasetController: {
    list: (...args: unknown[]) => mockList(...args),
    search: (...args: unknown[]) => mockSearch(...args),
    delete: (...args: unknown[]) => mockDelete(...args),
    getStats: (...args: unknown[]) => mockGetStats(...args),
    listVersions: vi.fn().mockResolvedValue([]),
    preview: vi.fn().mockResolvedValue({ headers: [], rows: [] }),
    export: vi.fn().mockResolvedValue(''),
  },
}))

describe('DatasetsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue([
      { id: '1', name: 'Shakespeare', row_count: 1000, size_bytes: 50000, created_at: '2025-01-01' },
    ])
    mockSearch.mockResolvedValue([])
    mockGetStats.mockResolvedValue({ total: 1, total_rows: 1000 })
  })

  it('renders page title', async () => {
    render(<DatasetsPage />)
    expect(screen.getByText('Datasets')).toBeTruthy()
  })

  it('loads and displays datasets', async () => {
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(screen.getByText('Shakespeare')).toBeTruthy()
    })
  })

  it('shows empty state when no datasets', async () => {
    mockList.mockResolvedValue([])
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(screen.getByText(/no datasets/i)).toBeTruthy()
    })
  })
})
