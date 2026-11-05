import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import DatasetsPage from './page'
import { useQueryStore } from '@/lib/cache/client'

const mockList = vi.fn()
const mockSearch = vi.fn()
const mockDelete = vi.fn()
const mockGetStats = vi.fn()

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() }),
}))

vi.mock('@/vite/next-compat/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/lib/dataset-controller', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/dataset-controller')>()
  return {
    ...actual,
    datasetController: {
      list: (...args: unknown[]) => mockList(...args),
      search: (...args: unknown[]) => mockSearch(...args),
      delete: (...args: unknown[]) => mockDelete(...args),
      getStats: (...args: unknown[]) => mockGetStats(...args),
      listVersions: vi.fn().mockResolvedValue([]),
      preview: vi.fn().mockResolvedValue({ headers: [], rows: [] }),
      export: vi.fn().mockResolvedValue(''),
    },
  }
})

describe('DatasetsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // useDatasetList caches by key (30s staleTime) — reset between tests.
    useQueryStore.setState({ cache: {}, fetchingKeys: new Set() })
    mockList.mockResolvedValue([
      {
        id: '1',
        name: 'Shakespeare',
        row_count: 1000,
        size_bytes: 50000,
        created_at: '2025-01-01',
      },
    ])
    mockSearch.mockResolvedValue([])
    mockGetStats.mockResolvedValue({ total: 1, total_rows: 1000 })
  })

  it('renders page title', async () => {
    render(<DatasetsPage />)
    expect(screen.getByText('My Files')).toBeTruthy()
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
      expect(screen.getByText(/no files yet|no datasets/i)).toBeTruthy()
    })
  })
})

describe('DatasetsPage structured listing (card 20261104_060023)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useQueryStore.setState({ cache: {}, fetchingKeys: new Set() })
    mockSearch.mockResolvedValue([])
    mockGetStats.mockResolvedValue({ total: 2, total_rows: 1000 })
  })

  it('groups system entries out of the main file list', async () => {
    mockList.mockResolvedValue([
      { id: 'ds1', name: 'Shakespeare', size: 50000, created_at: '2025-01-01', kind: 'dataset' },
      { id: 'sys1', name: 'Auth Mogdb', size: 0, created_at: '2025-01-01', kind: 'system' },
    ])
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(screen.getByText('Shakespeare')).toBeTruthy()
    })
    // system entry is not presented as a file in the main list...
    expect(screen.queryByText('Auth Mogdb')).toBeNull()
    // ...but is reachable via an explicit, labeled group control
    expect(screen.getByText(/system|storage entries/i)).toBeTruthy()
  })

  it('keeps untagged legacy entries in the main list', async () => {
    mockList.mockResolvedValue([
      { id: 'legacy', name: 'Legacy Set', size: 10, created_at: '2025-01-01' },
    ])
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(screen.getByText('Legacy Set')).toBeTruthy()
    })
    expect(screen.queryByText(/system|storage entries/i)).toBeNull()
  })

  it('labels corpusless entries instead of showing a bare size', async () => {
    mockList.mockResolvedValue([
      { id: 'hollow', name: 'Hollow', size: 0, created_at: '2025-01-01', kind: 'dataset', has_corpus: false },
      { id: 'full', name: 'Full Set', size: 900, created_at: '2025-01-01', kind: 'dataset', has_corpus: true, samples: 12 },
    ])
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(screen.getByText('Full Set')).toBeTruthy()
    })
    expect(screen.getByText('Hollow')).toBeTruthy()
    expect(screen.getByText(/no corpus/i)).toBeTruthy()
  })
})
