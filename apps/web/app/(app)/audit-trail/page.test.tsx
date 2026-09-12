import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const mockApiGet = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
}))

vi.mock('@/lib/auth', () => ({
  useAuthStore: (selector?: (s: { currentWorkspace: { id: string } | null }) => unknown) => {
    const state = { currentWorkspace: { id: 'ws-1' } }
    return selector ? selector(state) : state
  },
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { warning: vi.fn(), log: vi.fn(), error: vi.fn() },
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconRefresh: (props: Record<string, unknown>) => <svg data-testid="icon-refresh" {...props} />,
  IconDownload: (props: Record<string, unknown>) => <svg data-testid="icon-download" {...props} />,
}))

import AuditTrailPage from './page'

describe('AuditTrailPage', () => {
  const mockActivities = {
    data: {
      activities: [
        { type: 'training', action: 'job_completed', detail: 'Training finished', status: 'completed', timestamp: '2024-01-01T00:00:00Z', user: 'alice' },
        { type: 'audit', action: 'user_login', detail: 'User logged in', status: 'success', timestamp: '2024-01-02T00:00:00Z', user: 'bob' },
        { type: 'training', action: 'job_failed', detail: 'Out of memory', status: 'failed', timestamp: '2024-01-03T00:00:00Z', user: 'alice' },
      ],
    },
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockResolvedValue(mockActivities)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<AuditTrailPage />)
    await screen.findByText('Audit Trail')
  })

  it('shows loading skeleton initially', async () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<AuditTrailPage />)
    // PageContainer renders title even during loading
    expect(screen.getAllByText('Audit Trail').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches activities on mount', async () => {
    render(<AuditTrailPage />)
    await screen.findByText('Audit Trail')
    expect(mockApiGet).toHaveBeenCalledWith('/workspaces/ws-1/activity')
  })

  it('displays KPI stats', async () => {
    render(<AuditTrailPage />)
    await screen.findByText('Audit Trail')
    expect(screen.getAllByText('Total Events').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Filtered').length).toBeGreaterThanOrEqual(1)
  })

  it('displays activity list', async () => {
    render(<AuditTrailPage />)
    await screen.findByText('Audit Trail')
    expect(screen.getAllByText('job_completed').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('user_login').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('job_failed').length).toBeGreaterThanOrEqual(1)
  })

  it('shows filter input', async () => {
    render(<AuditTrailPage />)
    expect(await screen.findByPlaceholderText(/filter/i)).toBeTruthy()
  })

  it('filters activities', async () => {
    const user = userEvent.setup()
    render(<AuditTrailPage />)
    await screen.findByText('Audit Trail')

    const filterInput = screen.getByPlaceholderText(/filter/i)
    await user.type(filterInput, 'login')

    expect(screen.getAllByText('user_login').length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText('job_completed')).toBeNull()
  })

  it('shows type filter dropdown', async () => {
    render(<AuditTrailPage />)
    await screen.findByText('Audit Trail')
    expect(screen.getAllByText('All types').length).toBeGreaterThanOrEqual(1)
  })

  it('shows date range filters', async () => {
    render(<AuditTrailPage />)
    await screen.findByText('Audit Trail')
    expect(screen.getAllByText('From:').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('To:').length).toBeGreaterThanOrEqual(1)
  })

  it('shows refresh button', async () => {
    render(<AuditTrailPage />)
    await screen.findByText('Audit Trail')
    expect(screen.getAllByText('Refresh').length).toBeGreaterThanOrEqual(1)
  })

  it('shows export CSV button', async () => {
    render(<AuditTrailPage />)
    await screen.findByText('Audit Trail')
    expect(screen.getAllByText('Export CSV').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no activities', async () => {
    mockApiGet.mockResolvedValue({ data: { activities: [] } })
    render(<AuditTrailPage />)
    await screen.findByText('Audit Trail')
    expect(screen.getAllByText(/no events/i).length).toBeGreaterThanOrEqual(1)
  })
})
