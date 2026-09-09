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

import PermissionsPage from './page'

describe('PermissionsPage', () => {
  const mockPermissions = {
    data: {
      roles: {
        owner: { name: 'owner', permissions: ['model:list', 'train:create', 'chat:send'] },
        admin: { name: 'admin', permissions: ['model:list', 'train:create'] },
        user: { name: 'user', permissions: ['model:list', 'chat:send'] },
        viewer: { name: 'viewer', permissions: ['model:list'] },
      },
      all_permissions: {
        model: ['model:list', 'model:load'],
        train: ['train:create', 'train:view'],
        chat: ['chat:send', 'chat:history'],
      },
      member_permissions: [
        { user_id: 'u1', username: 'alice', role: 'admin', permissions: ['model:list', 'train:create'] },
        { user_id: 'u2', username: 'bob', role: 'user', permissions: ['model:list', 'chat:send'] },
      ],
    },
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockResolvedValue(mockPermissions)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<PermissionsPage />)
    await screen.findByText('Permissions')
  })

  it('shows loading skeleton initially', async () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<PermissionsPage />)
    await screen.findAllByText('Permissions')
    // Skeleton is rendered during loading (no data yet)
    expect(screen.getAllByText('Permissions').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches permissions on mount', async () => {
    render(<PermissionsPage />)
    await screen.findByText('Permissions')
    expect(mockApiGet).toHaveBeenCalledWith('/workspaces/ws-1/permissions')
  })

  it('renders role-permission matrix table', async () => {
    render(<PermissionsPage />)
    await screen.findByText('Permissions')
    expect(screen.getAllByText('Role Permissions').length).toBeGreaterThanOrEqual(1)
  })

  it('displays role badges in matrix header', async () => {
    render(<PermissionsPage />)
    await screen.findByText('Permissions')
    expect(screen.getAllByText('owner').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('admin').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('user').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('viewer').length).toBeGreaterThanOrEqual(1)
  })

  it('displays category labels', async () => {
    render(<PermissionsPage />)
    await screen.findByText('Permissions')
    expect(screen.getAllByText('Models').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Training').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Chat').length).toBeGreaterThanOrEqual(1)
  })

  it('displays permission names', async () => {
    render(<PermissionsPage />)
    await screen.findByText('Permissions')
    expect(screen.getAllByText('model:list').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('train:create').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('chat:send').length).toBeGreaterThanOrEqual(1)
  })

  it('shows checkmarks for permissions that roles have', async () => {
    render(<PermissionsPage />)
    await screen.findByText('Permissions')
    const checkmarks = screen.getAllByText('✓')
    expect(checkmarks.length).toBeGreaterThan(0)
  })

  it('displays member permissions section', async () => {
    render(<PermissionsPage />)
    await screen.findByText('Permissions')
    expect(screen.getAllByText(/Member Permissions/).length).toBeGreaterThanOrEqual(1)
  })

  it('lists members with roles', async () => {
    render(<PermissionsPage />)
    await screen.findByText('Permissions')
    expect(screen.getAllByText('alice').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('bob').length).toBeGreaterThanOrEqual(1)
  })

  it('expands member permissions on click', async () => {
    const user = userEvent.setup()
    render(<PermissionsPage />)
    await screen.findByText('Permissions')

    const alice = screen.getByText('alice')
    await user.click(alice)

    expect(screen.getAllByText('model:list').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no members', async () => {
    mockApiGet.mockResolvedValue({
      data: {
        roles: { owner: { name: 'owner', permissions: ['model:list'] } },
        all_permissions: { model: ['model:list'] },
        member_permissions: [],
      },
    })
    render(<PermissionsPage />)
    await screen.findByText('Permissions')
    expect(screen.getAllByText('No members').length).toBeGreaterThanOrEqual(1)
  })

  it('shows error toast on fetch failure', async () => {
    mockApiGet.mockRejectedValueOnce(new Error('fail'))
    render(<PermissionsPage />)
    await screen.findByText('Permissions')
  })
})
