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
  IconPlus: (props: Record<string, unknown>) => <svg data-testid="icon-plus" {...props} />,
  IconTrash: (props: Record<string, unknown>) => <svg data-testid="icon-trash" {...props} />,
  IconRefresh: (props: Record<string, unknown>) => <svg data-testid="icon-refresh" {...props} />,
}))

import ApiKeysPage from './page'

describe('ApiKeysPage', () => {
  const mockKeys = {
    data: {
      keys: [
        { id: 'k1', name: 'Production Key', key_hash: 'sk-abc...xyz', scopes: ['*'], created_at: 1700000000, revoked: false, workspace_id: 'ws-1', user_id: 'u1' },
        { id: 'k2', name: 'Dev Key', key_hash: 'sk-def...uvw', scopes: ['chat:send'], created_at: 1700100000, expires_at: 1700200000, revoked: false, workspace_id: 'ws-1', user_id: 'u1' },
        { id: 'k3', name: 'Old Key', key_hash: 'sk-old...key', scopes: ['*'], created_at: 1699900000, revoked: true, workspace_id: 'ws-1', user_id: 'u1' },
      ],
      count: 3,
    },
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockResolvedValue(mockKeys)
    mockApiPost.mockResolvedValue({ data: { key: 'sk-new-key-value', id: 'k4', name: 'New Key' } })
    mockApiDelete.mockResolvedValue({})
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<ApiKeysPage />)
    await screen.findByText('API Keys')
  })

  it('shows loading skeleton initially', () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<ApiKeysPage />)
    expect(screen.queryByText('API Keys')).toBeNull()
  })

  it('fetches keys on mount', async () => {
    render(<ApiKeysPage />)
    await screen.findByText('API Keys')
    expect(mockApiGet).toHaveBeenCalledWith('/security/keys')
  })

  it('displays active keys', async () => {
    render(<ApiKeysPage />)
    await screen.findByText('API Keys')
    expect(screen.getAllByText('Production Key').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Dev Key').length).toBeGreaterThanOrEqual(1)
  })

  it('displays revoked keys section', async () => {
    render(<ApiKeysPage />)
    await screen.findByText('API Keys')
    expect(screen.getAllByText('Revoked Keys').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Old Key').length).toBeGreaterThanOrEqual(1)
  })

  it('shows KPI stats', async () => {
    render(<ApiKeysPage />)
    await screen.findByText('API Keys')
    expect(screen.getAllByText('Active Keys').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Revoked').length).toBeGreaterThanOrEqual(1)
  })

  it('shows create form', async () => {
    render(<ApiKeysPage />)
    await screen.findByText('API Keys')
    expect(screen.getAllByText('Create API Key').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByPlaceholderText('Key name')).toBeTruthy()
  })

  it('creates a new key', async () => {
    const user = userEvent.setup()
    render(<ApiKeysPage />)
    await screen.findByText('API Keys')

    const input = screen.getByPlaceholderText('Key name')
    await user.type(input, 'New Key')
    const createBtn = screen.getByText('Create')
    await user.click(createBtn)

    expect(mockApiPost).toHaveBeenCalledWith('/security/keys', { name: 'New Key' })
  })

  it('shows new key after creation', async () => {
    const user = userEvent.setup()
    render(<ApiKeysPage />)
    await screen.findByText('API Keys')

    const input = screen.getByPlaceholderText('Key name')
    await user.type(input, 'New Key')
    await user.click(screen.getByText('Create'))

    expect(screen.getAllByText('New API Key').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('sk-new-key-value')).toBeTruthy()
  })

  it('shows empty state when no active keys', async () => {
    mockApiGet.mockResolvedValue({ data: { keys: [], count: 0 } })
    render(<ApiKeysPage />)
    await screen.findByText('API Keys')
    expect(screen.getAllByText('No active API keys.').length).toBeGreaterThanOrEqual(1)
  })

  it('shows key hash and creation date', async () => {
    render(<ApiKeysPage />)
    await screen.findByText('API Keys')
    expect(screen.getAllByText('sk-abc...xyz').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('sk-def...uvw').length).toBeGreaterThanOrEqual(1)
  })

  it('shows expiration for keys with expires_at', async () => {
    render(<ApiKeysPage />)
    await screen.findByText('API Keys')
    expect(screen.getAllByText(/expires/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows revoke button for active keys', async () => {
    render(<ApiKeysPage />)
    await screen.findByText('Production Key')
    const revokeButtons = screen.getAllByTitle('Revoke key')
    expect(revokeButtons.length).toBeGreaterThanOrEqual(1)
  })
})
