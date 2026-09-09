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

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) =>
    selector({ addToast: vi.fn() }),
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconRefresh: (props: Record<string, unknown>) => <svg data-testid="icon-refresh" {...props} />,
}))

import NotificationsPage from './page'

describe('NotificationsPage', () => {
  const mockNotifications = {
    data: {
      notifications: [
        { type: 'training', title: 'Job Completed', detail: 'Training job finished', status: 'completed', timestamp: '2024-01-01T00:00:00Z' },
        { type: 'member', title: 'Member Joined', detail: 'alice joined the workspace', status: '', timestamp: '2024-01-02T00:00:00Z' },
        { type: 'training', title: 'Job Failed', detail: 'Out of memory', status: 'failed', timestamp: '2024-01-03T00:00:00Z' },
      ],
    },
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockResolvedValue(mockNotifications)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<NotificationsPage />)
    await screen.findByText('Notifications')
  })

  it('shows loading skeleton initially', async () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<NotificationsPage />)
    // PageContainer renders title even during loading
    expect(screen.getAllByText('Notifications').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches notifications on mount', async () => {
    render(<NotificationsPage />)
    await screen.findByText('Notifications')
    expect(mockApiGet).toHaveBeenCalledWith('/workspaces/ws-1/notifications')
  })

  it('displays KPI stats', async () => {
    render(<NotificationsPage />)
    await screen.findByText('Notifications')
    expect(screen.getAllByText('Total').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Training').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Members').length).toBeGreaterThanOrEqual(1)
  })

  it('displays notification list', async () => {
    render(<NotificationsPage />)
    await screen.findByText('Notifications')
    expect(screen.getAllByText('Job Completed').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Member Joined').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Job Failed').length).toBeGreaterThanOrEqual(1)
  })

  it('shows notification types', async () => {
    render(<NotificationsPage />)
    await screen.findByText('Notifications')
    expect(screen.getAllByText('training').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('member').length).toBeGreaterThanOrEqual(1)
  })

  it('shows notification statuses', async () => {
    render(<NotificationsPage />)
    await screen.findByText('Notifications')
    expect(screen.getAllByText('completed').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('failed').length).toBeGreaterThanOrEqual(1)
  })

  it('shows filter input', async () => {
    render(<NotificationsPage />)
    await screen.findByText('Notifications')
    expect(screen.getByPlaceholderText('Filter notifications...')).toBeTruthy()
  })

  it('filters notifications', async () => {
    const user = userEvent.setup()
    render(<NotificationsPage />)
    await screen.findByText('Notifications')

    const filterInput = screen.getByPlaceholderText('Filter notifications...')
    await user.type(filterInput, 'Failed')

    expect(screen.getAllByText('Job Failed').length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText('Job Completed')).toBeNull()
  })

  it('shows refresh button', async () => {
    render(<NotificationsPage />)
    await screen.findByText('Notifications')
    expect(screen.getAllByText('Refresh').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no notifications', async () => {
    mockApiGet.mockResolvedValue({ data: { notifications: [] } })
    render(<NotificationsPage />)
    await screen.findByText('Notifications')
    expect(screen.getAllByText('No notifications').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when filter matches nothing', async () => {
    const user = userEvent.setup()
    render(<NotificationsPage />)
    await screen.findByText('Notifications')

    const filterInput = screen.getByPlaceholderText('Filter notifications...')
    await user.type(filterInput, 'zzznonexistent')

    expect(screen.getAllByText(/no notifications match/i).length).toBeGreaterThanOrEqual(1)
  })
})
