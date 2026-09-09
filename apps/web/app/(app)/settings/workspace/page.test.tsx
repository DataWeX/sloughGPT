import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const mockApiGet = vi.fn()
const mockApiPut = vi.fn()
const mockApiDelete = vi.fn()
const mockApiPost = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPut: (...args: unknown[]) => mockApiPut(...args),
  apiDelete: (...args: unknown[]) => mockApiDelete(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
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

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (key: string) => key }),
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconTrash: (props: Record<string, unknown>) => <svg data-testid="icon-trash" {...props} />,
}))

import WorkspaceSettingsPage from './page'

describe('WorkspaceSettingsPage', () => {
  const mockSettings = {
    workspace_id: 'ws-1',
    name: 'Test Workspace',
    description: 'A test workspace',
    default_model: 'llama-3.1-8b',
    data_retention_days: 90,
    training_retention_days: 30,
    audit_retention_days: 60,
    dataset_retention_days: 45,
    max_members: 50,
    allow_sharing: true,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-06-01T00:00:00Z',
    member_count: 5,
  }

  const mockUsage = {
    members: { total: 5, admins: 1, users: 3, viewers: 1 },
    training: { total: 12, running: 2, completed: 8, failed: 2, total_minutes: 120 },
    datasets: { total: 7 },
    knowledge: { total: 3 },
    api_keys: { total: 4 },
  }

  const mockHealth = {
    status: 'healthy',
    checks: [
      { name: 'database', detail: 'connected' },
      { name: 'disk', detail: 'ok' },
    ],
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.spyOn(Storage.prototype, 'getItem').mockReturnValue(null)
    mockApiGet
      .mockResolvedValueOnce({ data: mockSettings })
      .mockResolvedValueOnce({ data: mockUsage })
      .mockResolvedValueOnce({ data: mockHealth })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Test Workspace — Settings')
  })

  it('shows loading skeleton initially', () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<WorkspaceSettingsPage />)
    expect(screen.queryByText('Test Workspace — Settings')).toBeNull()
  })

  it('fetches settings, usage, and health on mount', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Test Workspace — Settings')
    expect(mockApiGet).toHaveBeenCalledWith('/workspaces/ws-1/settings')
    expect(mockApiGet).toHaveBeenCalledWith('/workspaces/ws-1/usage')
    expect(mockApiGet).toHaveBeenCalledWith('/workspaces/ws-1/health')
  })

  it('displays KPI stats from usage', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Test Workspace — Settings')
    expect(screen.getAllByText('Members').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Training Jobs').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Datasets').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Knowledge').length).toBeGreaterThanOrEqual(1)
  })

  it('populates form fields from settings', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Test Workspace — Settings')
    expect(screen.getByDisplayValue('Test Workspace')).toBeTruthy()
    expect(screen.getByDisplayValue('A test workspace')).toBeTruthy()
    expect(screen.getByDisplayValue('llama-3.1-8b')).toBeTruthy()
  })

  it('renders retention sliders', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Test Workspace — Settings')
    expect(screen.getAllByText(/Default Retention/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Training:/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Audit Logs:/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Datasets:/).length).toBeGreaterThanOrEqual(1)
  })

  it('renders allow sharing switch', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Test Workspace — Settings')
    expect(screen.getAllByText('Allow Sharing').length).toBeGreaterThanOrEqual(1)
  })

  it('renders health status', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Test Workspace — Settings')
    expect(screen.getAllByText('healthy').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('database').length).toBeGreaterThanOrEqual(1)
  })

  it('renders data retention card', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Test Workspace — Settings')
    expect(screen.getAllByText('Data Retention').length).toBeGreaterThanOrEqual(1)
  })

  it('renders danger zone', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Test Workspace — Settings')
    expect(screen.getAllByText('Danger Zone').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Delete Workspace').length).toBeGreaterThanOrEqual(1)
  })

  it('shows save button', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Test Workspace — Settings')
    expect(screen.getAllByText('Save Changes').length).toBeGreaterThanOrEqual(1)
  })

  it('shows clone button', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Test Workspace — Settings')
    expect(screen.getAllByText('Clone Workspace').length).toBeGreaterThanOrEqual(1)
  })

  it('shows cleanup button', async () => {
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Test Workspace — Settings')
    expect(screen.getAllByText('Run Cleanup Now').length).toBeGreaterThanOrEqual(1)
  })

  it('saves settings on button click', async () => {
    const user = userEvent.setup()
    mockApiPut.mockResolvedValueOnce({})
    render(<WorkspaceSettingsPage />)
    await screen.findByText('Test Workspace — Settings')

    const nameInput = screen.getByDisplayValue('Test Workspace')
    await user.clear(nameInput)
    await user.type(nameInput, 'Updated Workspace')

    const saveBtn = screen.getByText('Save Changes')
    await user.click(saveBtn)
    expect(mockApiPut).toHaveBeenCalledWith('/workspaces/ws-1/settings', expect.objectContaining({
      name: 'Updated Workspace',
    }))
  })

  it('shows no workspace message when settings are null', async () => {
    mockApiGet.mockReset()
    mockApiGet.mockResolvedValue(null)
    render(<WorkspaceSettingsPage />)
    await screen.findByText('No workspace selected.')
  })
})
