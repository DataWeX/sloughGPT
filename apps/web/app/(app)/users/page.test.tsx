import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const mockApiGet = vi.fn()
const mockApiPost = vi.fn()
const mockApiPut = vi.fn()
const mockApiDelete = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiPut: (...args: unknown[]) => mockApiPut(...args),
  apiDelete: (...args: unknown[]) => mockApiDelete(...args),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) =>
    selector({ addToast: vi.fn() }),
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { warning: vi.fn(), log: vi.fn(), error: vi.fn() },
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconPlus: (props: Record<string, unknown>) => <svg data-testid="icon-plus" {...props} />,
  IconTrash: (props: Record<string, unknown>) => <svg data-testid="icon-trash" {...props} />,
  IconRefresh: (props: Record<string, unknown>) => <svg data-testid="icon-refresh" {...props} />,
}))

import UsersAdminPage from './page'

describe('UsersAdminPage', () => {
  const mockUsers = {
    data: [
      { id: 'u1', username: 'alice', email: 'alice@test.com', role: 'admin', status: 'active', display_name: 'Alice', tenant_id: 't1', created_at: '2024-01-01T00:00:00Z', last_login_at: '2024-06-01T00:00:00Z' },
      { id: 'u2', username: 'bob', email: 'bob@test.com', role: 'user', status: 'active', display_name: 'Bob', tenant_id: 't1', created_at: '2024-02-01T00:00:00Z', last_login_at: '2024-05-01T00:00:00Z' },
      { id: 'u3', username: 'carol', email: 'carol@test.com', role: 'viewer', status: 'inactive', display_name: 'Carol', tenant_id: 't1', created_at: '2024-03-01T00:00:00Z', last_login_at: '' },
    ],
    meta: { total: 3 },
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockResolvedValue(mockUsers)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<UsersAdminPage />)
    await screen.findByText('Users')
  })

  it('shows loading skeleton initially', () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<UsersAdminPage />)
    expect(screen.getAllByText('Loading').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches users on mount', async () => {
    render(<UsersAdminPage />)
    await screen.findByText('Users')
    expect(mockApiGet).toHaveBeenCalledWith('/users')
  })

  it('displays KPI stats', async () => {
    render(<UsersAdminPage />)
    await screen.findByText('Users')
    expect(screen.getAllByText('Total Users').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Active').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Admins').length).toBeGreaterThanOrEqual(1)
  })

  it('displays user list', async () => {
    render(<UsersAdminPage />)
    await screen.findByText('Users')
    expect(screen.getAllByText('alice').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('bob').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('carol').length).toBeGreaterThanOrEqual(1)
  })

  it('shows user roles and statuses', async () => {
    render(<UsersAdminPage />)
    await screen.findByText('Users')
    expect(screen.getAllByText('admin').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('active').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('inactive').length).toBeGreaterThanOrEqual(1)
  })

  it('shows create user form', async () => {
    render(<UsersAdminPage />)
    await screen.findByText('Users')
    expect(screen.getAllByText('Create User').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByPlaceholderText('Username')).toBeTruthy()
    expect(screen.getByPlaceholderText('Email')).toBeTruthy()
  })

  it('shows edit button for users', async () => {
    render(<UsersAdminPage />)
    await screen.findByText('Users')
    expect(screen.getAllByText('Edit').length).toBeGreaterThanOrEqual(1)
  })

  it('shows refresh button', async () => {
    render(<UsersAdminPage />)
    await screen.findByText('Users')
    expect(screen.getAllByText('Refresh').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no users', async () => {
    mockApiGet.mockResolvedValue({ data: [], meta: { total: 0 } })
    render(<UsersAdminPage />)
    await screen.findByText('Users')
    expect(screen.getAllByText(/no users/i).length).toBeGreaterThanOrEqual(1)
  })
})
