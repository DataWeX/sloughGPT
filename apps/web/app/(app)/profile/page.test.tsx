import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const mockApiGet = vi.fn()
const mockApiPut = vi.fn()
const mockApiPost = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPut: (...args: unknown[]) => mockApiPut(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
}))

vi.mock('@/lib/auth', () => ({
  useAuthStore: (selector?: (s: { currentWorkspace: { id: string } | null; user: { id: string; email: string } | null; setUser: (u: unknown) => void }) => unknown) => {
    const state = { currentWorkspace: { id: 'ws-1' }, user: { id: 'u1', email: 'test@test.com' }, setUser: vi.fn() }
    return selector ? selector(state) : state
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) =>
    selector({ addToast: vi.fn() }),
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { warning: vi.fn(), log: vi.fn(), error: vi.fn() },
}))

import ProfilePage from './page'

describe('ProfilePage', () => {
  const mockProfile = {
    data: {
      id: 'u1',
      username: 'alice',
      email: 'alice@test.com',
      role: 'admin',
      status: 'active',
      display_name: 'Alice',
      tenant_id: 't1',
      created_at: '2024-01-01T00:00:00Z',
      last_login_at: '2024-06-01T00:00:00Z',
    },
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockResolvedValue(mockProfile)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<ProfilePage />)
    await screen.findByText('Profile')
  })

  it('shows loading skeleton initially', () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<ProfilePage />)
    expect(screen.getAllByText('Loading').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches profile on mount', async () => {
    render(<ProfilePage />)
    await screen.findByText('Profile')
    expect(mockApiGet).toHaveBeenCalledWith('/users/me/profile')
  })

  it('displays KPI stats', async () => {
    render(<ProfilePage />)
    await screen.findByText('Profile')
    expect(screen.getAllByText('Username').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Role').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Status').length).toBeGreaterThanOrEqual(1)
  })

  it('displays profile information', async () => {
    render(<ProfilePage />)
    await screen.findByText('Profile')
    expect(screen.getAllByText('Profile Information').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByDisplayValue('alice')).toBeTruthy()
    expect(screen.getByDisplayValue('alice@test.com')).toBeTruthy()
  })

  it('displays change password form', async () => {
    render(<ProfilePage />)
    await screen.findByText('Profile')
    expect(screen.getAllByText('Change Password').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByPlaceholderText('Current password')).toBeTruthy()
    expect(screen.getByPlaceholderText('New password (min 8 characters)')).toBeTruthy()
  })

  it('displays account details', async () => {
    render(<ProfilePage />)
    await screen.findByText('Profile')
    expect(screen.getAllByText('Account Details').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('User ID:').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Tenant ID:').length).toBeGreaterThanOrEqual(1)
  })

  it('saves profile changes', async () => {
    const user = userEvent.setup()
    mockApiPut.mockResolvedValueOnce({ data: { email: 'new@test.com', display_name: 'New Name' } })
    render(<ProfilePage />)
    await screen.findByText('Profile')

    const emailInput = screen.getByDisplayValue('alice@test.com')
    await user.clear(emailInput)
    await user.type(emailInput, 'new@test.com')

    const saveBtn = screen.getByText('Save Changes')
    await user.click(saveBtn)
    expect(mockApiPut).toHaveBeenCalledWith('/users/me/profile', expect.objectContaining({ email: 'new@test.com' }))
  })
})
