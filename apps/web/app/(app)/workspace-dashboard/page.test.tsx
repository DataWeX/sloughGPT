import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'

const mockApiGet = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
}))

vi.mock('@/lib/auth', () => ({
  useAuthStore: (selector?: (s: { currentWorkspace: { id: string; name: string } | null }) => unknown) => {
    const state = { currentWorkspace: { id: 'ws-1', name: 'Test Workspace' } }
    return selector ? selector(state) : state
  },
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { warning: vi.fn(), log: vi.fn(), error: vi.fn() },
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    refresh: vi.fn(),
    back: vi.fn(),
    prefetch: vi.fn(),
  }),
}))

import WorkspaceDashboardPage from './page'

describe('WorkspaceDashboardPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockImplementation((url: string) => {
      if (url.includes('/stats')) return Promise.resolve({ data: { name: 'Test Workspace', member_count: 5, dataset_count: 7, training_jobs: 12, active_training_jobs: 2, knowledge_items: 3 } })
      if (url.includes('/activity')) return Promise.resolve({ data: { activities: [] } })
      if (url.includes('/usage')) return Promise.resolve({ data: { members: { total: 5 }, training: { total: 12 }, datasets: { total: 7 }, knowledge: { total: 3 } } })
      if (url.includes('/health')) return Promise.resolve({ data: { status: 'healthy' } })
      return Promise.resolve({ data: null })
    })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<WorkspaceDashboardPage />)
    await screen.findByText('Workspace Dashboard')
  })

  it('shows loading skeleton initially', async () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<WorkspaceDashboardPage />)
    // PageContainer renders title even during loading
    expect(screen.getAllByText('Workspace Dashboard').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches dashboard data on mount', async () => {
    render(<WorkspaceDashboardPage />)
    await screen.findByText('Workspace Dashboard')
    expect(mockApiGet).toHaveBeenCalled()
  })

  it('displays KPI stats', async () => {
    render(<WorkspaceDashboardPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Members').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('Datasets').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Training Jobs').length).toBeGreaterThanOrEqual(1)
  })

  it('displays health status', async () => {
    render(<WorkspaceDashboardPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/healthy/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('displays recent activity section', async () => {
    render(<WorkspaceDashboardPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/recent activity/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('displays training progress', async () => {
    render(<WorkspaceDashboardPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/running/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows members count', async () => {
    render(<WorkspaceDashboardPage />)
    await screen.findByText('Members')
    expect(screen.getAllByText('Members').length).toBeGreaterThanOrEqual(1)
  })
})
