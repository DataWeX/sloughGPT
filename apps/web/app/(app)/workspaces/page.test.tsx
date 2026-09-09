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

vi.mock('@/lib/auth', () => ({
  useAuthStore: (selector?: (s: { currentWorkspace: { id: string; name: string } | null; switchWorkspace: (id: string) => void }) => unknown) => {
    const state = { currentWorkspace: { id: 'ws-1', name: 'Current WS' }, switchWorkspace: vi.fn() }
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

vi.mock('@/components/icons/NavIcons', () => ({
  IconPlus: (props: Record<string, unknown>) => <svg data-testid="icon-plus" {...props} />,
  IconTrash: (props: Record<string, unknown>) => <svg data-testid="icon-trash" {...props} />,
}))

import WorkspacesPage from './page'

describe('WorkspacesPage', () => {
  const mockWorkspaces = {
    data: [
      { id: 'ws-1', name: 'Workspace A', description: 'First', tenant_id: 't1', role: 'owner', member_count: 5, created_at: '2024-01-01T00:00:00Z' },
      { id: 'ws-2', name: 'Workspace B', description: 'Second', tenant_id: 't1', role: 'member', member_count: 3, created_at: '2024-02-01T00:00:00Z' },
    ],
    meta: { total: 2 },
  }

  const mockMembers = {
    data: [
      { user_id: 'u1', username: 'alice', email: 'alice@test.com', role: 'admin' },
      { user_id: 'u2', username: 'bob', email: 'bob@test.com', role: 'member' },
    ],
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockImplementation((url: string) => {
      if (url === '/workspaces') return Promise.resolve(mockWorkspaces)
      if (url.includes('/members')) return Promise.resolve(mockMembers)
      return Promise.resolve({ data: [] })
    })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<WorkspacesPage />)
    await screen.findByText('Workspaces')
  })

  it('shows loading skeleton initially', () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<WorkspacesPage />)
    expect(screen.getAllByText('Loading').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches workspaces on mount', async () => {
    render(<WorkspacesPage />)
    await screen.findByText('Workspaces')
    expect(mockApiGet).toHaveBeenCalledWith('/workspaces')
  })

  it('displays KPI stats', async () => {
    render(<WorkspacesPage />)
    await screen.findByText('Workspaces')
    expect(screen.getAllByText('Workspaces').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Current').length).toBeGreaterThanOrEqual(1)
  })

  it('displays workspace list', async () => {
    render(<WorkspacesPage />)
    await screen.findByText('Workspaces')
    expect(screen.getAllByText('Workspace A').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Workspace B').length).toBeGreaterThanOrEqual(1)
  })

  it('shows create workspace form', async () => {
    render(<WorkspacesPage />)
    await screen.findByText('Workspaces')
    expect(screen.getAllByText('Create Workspace').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByPlaceholderText('Workspace name')).toBeTruthy()
  })

  it('shows import button', async () => {
    render(<WorkspacesPage />)
    await screen.findByText('Workspaces')
    expect(screen.getAllByText('Import from JSON').length).toBeGreaterThanOrEqual(1)
  })

  it('shows edit button for workspaces', async () => {
    render(<WorkspacesPage />)
    await screen.findByText('Workspaces')
    expect(screen.getAllByText('Edit').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no workspaces', async () => {
    mockApiGet.mockResolvedValue({ data: [], meta: { total: 0 } })
    render(<WorkspacesPage />)
    await screen.findByText('Workspaces')
    expect(screen.getAllByText(/no workspaces/i).length).toBeGreaterThanOrEqual(1)
  })
})
