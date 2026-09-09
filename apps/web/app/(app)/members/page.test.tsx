import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const mockApiGet = vi.fn()
const mockApiPost = vi.fn()
const mockApiDelete = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiDelete: (...args: unknown[]) => mockApiDelete(...args),
}))

vi.mock('@/lib/auth', () => ({
  useAuthStore: (selector?: (s: { currentWorkspace: { id: string } | null; switchWorkspace: (id: string) => void }) => unknown) => {
    const state = {
      currentWorkspace: { id: 'ws-1' },
      switchWorkspace: vi.fn(),
    }
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
  IconUpload: (props: Record<string, unknown>) => <svg data-testid="icon-upload" {...props} />,
}))

import MembersPage from './page'

describe('MembersPage', () => {
  const mockWorkspaces = {
    data: [
      { id: 'ws-1', name: 'Test Workspace', member_count: 3 },
      { id: 'ws-2', name: 'Other Workspace', member_count: 1 },
    ],
    meta: { total: 2 },
  }

  const mockMembers = {
    data: [
      { user_id: 'u1', username: 'alice', email: 'alice@test.com', role: 'admin', joined_at: '2024-01-01T00:00:00Z' },
      { user_id: 'u2', username: 'bob', email: 'bob@test.com', role: 'member', joined_at: '2024-02-01T00:00:00Z' },
      { user_id: 'u3', username: 'carol', email: 'carol@test.com', role: 'viewer', joined_at: '2024-03-01T00:00:00Z' },
    ],
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet
      .mockResolvedValueOnce(mockWorkspaces)
      .mockResolvedValue({ data: [] })
    mockApiPost.mockResolvedValue({})
    mockApiDelete.mockResolvedValue({})
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<MembersPage />)
    await screen.findByText('Members')
  })

  it('shows loading skeleton initially', () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<MembersPage />)
    expect(screen.getAllByText('Loading').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches workspaces on mount', async () => {
    render(<MembersPage />)
    await screen.findByText('Members')
    expect(mockApiGet).toHaveBeenCalledWith('/workspaces')
  })

  it('displays workspace selector', async () => {
    render(<MembersPage />)
    await screen.findByText('Members')
    expect(screen.getAllByText('Test Workspace').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Other Workspace').length).toBeGreaterThanOrEqual(1)
  })

  it('shows KPI stats after selecting workspace', async () => {
    render(<MembersPage />)
    await screen.findByText('Members')
    expect(screen.getAllByText('Total Members').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Admins').length).toBeGreaterThanOrEqual(1)
  })

  it('shows add member form', async () => {
    render(<MembersPage />)
    await screen.findByText('Members')
    expect(screen.getByPlaceholderText('User ID')).toBeTruthy()
    expect(screen.getAllByText('Add').length).toBeGreaterThanOrEqual(1)
  })

  it('shows invite form', async () => {
    render(<MembersPage />)
    await screen.findByText('Members')
    expect(screen.getAllByText(/invite/i).length).toBeGreaterThanOrEqual(1)
  })

  it('shows search input', async () => {
    render(<MembersPage />)
    await screen.findByText('Members')
    expect(screen.getByPlaceholderText(/search/i)).toBeTruthy()
  })

  it('shows bulk import button', async () => {
    render(<MembersPage />)
    await screen.findByText('Members')
    expect(screen.getAllByText(/bulk import/i).length).toBeGreaterThanOrEqual(1)
  })
})
