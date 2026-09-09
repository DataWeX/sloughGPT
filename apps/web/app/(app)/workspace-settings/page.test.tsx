import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const mockApiGet = vi.fn()
const mockApiPut = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPut: (...args: unknown[]) => mockApiPut(...args),
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

vi.mock('@/lib/dev-log', () => ({
  logger: { warning: vi.fn(), log: vi.fn(), error: vi.fn() },
}))

import WorkspaceSettingsPage from './page'

describe('WorkspaceSettingsPage', () => {
  const mockSettings = {
    data: {
      workspace_id: 'ws-1',
      name: 'Test Workspace',
      description: 'A test workspace',
      default_model: 'llama-3.2-3b',
      data_retention_days: 90,
      max_members: 50,
      allow_sharing: true,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-06-01T00:00:00Z',
    },
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockResolvedValue(mockSettings)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Workspace Settings')
  })

  it('shows loading skeleton initially', async () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<WorkspaceSettingsPage />)
    expect(screen.getAllByText('Workspace Settings').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches settings on mount', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Workspace Settings')
    expect(mockApiGet).toHaveBeenCalledWith('/workspaces/ws-1/settings')
  })

  it('displays general section', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Workspace Settings')
    expect(screen.getAllByText('General').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByDisplayValue('Test Workspace')).toBeTruthy()
  })

  it('displays defaults section', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Workspace Settings')
    expect(screen.getAllByText('Defaults').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByDisplayValue('llama-3.2-3b')).toBeTruthy()
  })

  it('displays limits section', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Workspace Settings')
    expect(screen.getAllByText('Limits').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Data Retention (days)').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Max Members').length).toBeGreaterThanOrEqual(1)
  })

  it('displays sharing section', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Workspace Settings')
    expect(screen.getAllByText('Sharing').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Allow data sharing').length).toBeGreaterThanOrEqual(1)
  })

  it('shows save button', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Workspace Settings')
    expect(screen.getAllByText('Save Settings').length).toBeGreaterThanOrEqual(1)
  })

  it('shows no changes initially', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Workspace Settings')
    expect(screen.getAllByText('No changes').length).toBeGreaterThanOrEqual(1)
  })

  it('saves settings', async () => {
    const user = userEvent.setup()
    mockApiPut.mockResolvedValueOnce({})
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Workspace Settings')

    const nameInput = screen.getByDisplayValue('Test Workspace')
    await user.clear(nameInput)
    await user.type(nameInput, 'Updated Name')

    const saveBtn = screen.getByText('Save Settings')
    await user.click(saveBtn)
    expect(mockApiPut).toHaveBeenCalledWith('/workspaces/ws-1/settings', expect.objectContaining({ name: 'Updated Name' }))
  })

  it('shows no workspace message when settings are null', async () => {
    mockApiGet.mockResolvedValue(null)
    render(<WorkspaceSettingsPage />)
    await screen.findByText(/no workspace selected/i)
  })
})
