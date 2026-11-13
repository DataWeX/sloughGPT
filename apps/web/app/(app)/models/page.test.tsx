import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'

// The models surface is the merged target of the v1 route map (card
// 20261101_154054): the Developer Models tab moved here and /model/[id]
// detail is composed as a FoldSection driven by ?id=.
const { mockReplace, mockPush, mockSearchParams, mockApiGet, mockRefetch } = vi.hoisted(() => ({
  mockReplace: vi.fn(),
  mockPush: vi.fn(),
  mockSearchParams: { get: vi.fn() },
  mockApiGet: vi.fn(),
  mockRefetch: vi.fn(),
}))

vi.mock('@/vite/next-compat/navigation', () => ({
  useRouter: () => ({ push: mockPush, replace: mockReplace }),
  useSearchParams: () => mockSearchParams,
  usePathname: () => '/models',
  useParams: () => ({}),
}))

vi.mock('@/lib/cache/api-hooks', () => ({
  useModels: () => ({ data: [{ id: 'gpt2' }], isLoading: false, refetch: mockRefetch }),
}))

vi.mock('@/hooks/useLiveStatus', () => ({
  useLiveStatus: () => ({
    healthLegacy: { model_loaded: true, model_type: 'gpt2' },
    connectionStatus: 'connected',
  }),
}))

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  authFetch: (url: string, init?: RequestInit) => globalThis.fetch(url, init),
}))

vi.mock('@/components/models/ModelCatalogCard', () => ({
  default: () => <div data-testid="model-catalog-card" />,
}))
vi.mock('@/components/models/ModelLineageCard', () => ({
  default: () => <div data-testid="model-lineage-card" />,
}))
vi.mock('@/components/models/QuantizationCard', () => ({
  default: () => <div data-testid="quantization-card" />,
}))
vi.mock('@/components/models/DownloadsCard', () => ({
  default: () => <div data-testid="downloads-card" />,
}))
vi.mock('@/components/models/EngineStatusCard', () => ({
  default: () => <div data-testid="engine-status-card" />,
}))

import ModelsPage from './page'

describe('ModelsPage (merged /models surface)', () => {
  beforeEach(() => {
    mockSearchParams.get.mockReturnValue(null)
    mockApiGet.mockResolvedValue({
      model_id: 'gpt2',
      status: 'ready',
      device: 'cpu',
      is_default: true,
      circuit_breaker: 'closed',
      warmup_completed: true,
      warmup_error: null,
      total_requests: 42,
    })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders the full models surface (catalog + engine cards), never redirects', () => {
    render(<ModelsPage />)
    expect(screen.getByTestId('model-catalog-card')).toBeTruthy()
    expect(screen.getByTestId('model-lineage-card')).toBeTruthy()
    expect(screen.getByTestId('quantization-card')).toBeTruthy()
    expect(screen.getByTestId('downloads-card')).toBeTruthy()
    expect(screen.getByTestId('engine-status-card')).toBeTruthy()
    // the old page was a client-side redirect to /developer — gone for good
    expect(mockReplace).not.toHaveBeenCalled()
  })

  it('does not fetch detail until a model is selected', async () => {
    render(<ModelsPage />)
    expect(screen.getByText('Model Details')).toBeTruthy()
    expect(mockApiGet).not.toHaveBeenCalled()
    expect(screen.queryByTestId('model-detail-loading')).toBeNull()
  })

  it('loads detail when ?id= is present and renders it', async () => {
    mockSearchParams.get.mockImplementation((k: string) => (k === 'id' ? 'gpt2' : null))
    render(<ModelsPage />)
    expect(mockApiGet).toHaveBeenCalledWith('/registry/models/gpt2')
    // explicit loading boolean — the skeleton shows while the fetch is pending
    expect(screen.getByTestId('model-detail-loading')).toBeTruthy()
    await waitFor(() => expect(screen.getByText('Total Requests')).toBeTruthy())
    expect(screen.getByText('42')).toBeTruthy()
    expect(screen.queryByTestId('model-detail-loading')).toBeNull()
  })

  it('keeps the detail folded (no fetch, no skeleton) when ?id= is cleared', () => {
    mockSearchParams.get.mockReturnValue(null)
    render(<ModelsPage />)
    expect(mockApiGet).not.toHaveBeenCalled()
    expect(screen.queryByTestId('model-detail-loading')).toBeNull()
    expect(screen.queryByTestId('model-details-body')).toBeNull()
  })
})
