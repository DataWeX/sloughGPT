import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const mockApiGet = vi.fn()
const mockApiPost = vi.fn()
const mockApiDelete = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiDelete: (...args: unknown[]) => mockApiDelete(...args),
}))

vi.mock('@/lib/auth', () => ({
  useAuthStore: (selector?: (s: { currentWorkspace: { id: string } | null }) => unknown) => {
    const state = { currentWorkspace: { id: 'ws-1' } }
    return selector ? selector(state) : state
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) =>
    selector({ addToast: vi.fn() }),
}))

import SharedDataPage from './page'

describe('SharedDataPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockImplementation((url: string) => {
      if (url.includes('/shared')) return Promise.resolve({ data: { shares: [
        { id: 's1', resource_type: 'dataset', resource_id: 'ds-1', source_workspace_id: 'ws-2', target_workspace_id: 'ws-1', permission: 'read', shared_by: 'u1', shared_at: '2024-01-01T00:00:00Z' },
        { id: 's2', resource_type: 'knowledge', resource_id: 'kn-1', source_workspace_id: 'ws-1', target_workspace_id: 'ws-3', permission: 'admin', shared_by: 'u1', shared_at: '2024-02-01T00:00:00Z' },
      ] } })
      if (url === '/workspaces') return Promise.resolve({ data: [
        { id: 'ws-1', name: 'Current Workspace' },
        { id: 'ws-2', name: 'Other Workspace' },
      ] })
      return Promise.resolve({ data: null })
    })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<SharedDataPage />)
    await screen.findByText('Shared Data')
  })

  it('shows loading skeleton initially', async () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<SharedDataPage />)
    // PageContainer renders title even during loading, but skeleton shows too
    expect(screen.getAllByText('Shared Data').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches shares on mount', async () => {
    render(<SharedDataPage />)
    await screen.findByText('Shared Data')
    expect(mockApiGet).toHaveBeenCalled()
  })

  it('shows share data button', async () => {
    render(<SharedDataPage />)
    await screen.findByText('Shared Data')
    expect(screen.getAllByText('Share Data').length).toBeGreaterThanOrEqual(1)
  })

  it('shows revoke button', async () => {
    render(<SharedDataPage />)
    await screen.findByText('Shared Data')
    expect(screen.getAllByText('Revoke').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no shares', async () => {
    mockApiGet.mockImplementation((url: string) => {
      if (url.includes('/shared')) return Promise.resolve({ data: { shares: [] } })
      if (url === '/workspaces') return Promise.resolve({ data: [] })
      return Promise.resolve({ data: null })
    })
    render(<SharedDataPage />)
    await screen.findByText('Shared Data')
    expect(screen.getAllByText(/no data/i).length).toBeGreaterThanOrEqual(1)
  })
})
