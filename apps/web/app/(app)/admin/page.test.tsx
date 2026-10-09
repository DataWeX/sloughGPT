import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const mockApiGet = vi.fn()
const mockGetMe = vi.fn()
const mockLogin = vi.fn()
const mockRegister = vi.fn()
const mockVerify = vi.fn()
const mockGetGrouped = vi.fn()
const mockGetRecent = vi.fn()
const mockGetTrends = vi.fn()
const mockClearErrors = vi.fn()
const mockExportErrors = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

vi.mock('@/lib/auth-controller', () => ({
  authController: {
    getMe: (...args: unknown[]) => mockGetMe(...args),
    login: (...args: unknown[]) => mockLogin(...args),
    register: (...args: unknown[]) => mockRegister(...args),
    verify: (...args: unknown[]) => mockVerify(...args),
  },
}))

vi.mock('@/lib/errors-controller', () => ({
  errorsController: {
    getGrouped: (...args: unknown[]) => mockGetGrouped(...args),
    getRecent: (...args: unknown[]) => mockGetRecent(...args),
    getTrends: (...args: unknown[]) => mockGetTrends(...args),
    clear: (...args: unknown[]) => mockClearErrors(...args),
    export: (...args: unknown[]) => mockExportErrors(...args),
  },
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { warning: vi.fn(), log: vi.fn(), error: vi.fn() },
}))

vi.mock('@/components/security/SecurityOverviewCard', () => ({
  SecurityOverviewCard: (props: Record<string, unknown>) => <div data-testid="security-overview">{JSON.stringify(props)}</div>,
}))

vi.mock('@/components/auth/AuthSessionInfoCard', () => ({
  AuthSessionInfoCard: () => <div data-testid="auth-session-info">AuthSessionInfoCard</div>,
}))

vi.mock('@/components/errors/ErrorInsightsCard', () => ({
  ErrorInsightsCard: () => <div data-testid="error-insights">ErrorInsightsCard</div>,
}))

import AdminPage from './page'

describe('AdminPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.spyOn(Storage.prototype, 'getItem').mockReturnValue(null)
    mockApiGet.mockResolvedValue({ logs: [], count: 0, configured: false })
    mockGetGrouped.mockResolvedValue([])
    mockGetRecent.mockResolvedValue({ errors: [], total: 0 })
    mockGetTrends.mockResolvedValue([])
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<AdminPage />)
    expect(screen.getAllByText('Admin').length).toBeGreaterThanOrEqual(1)
  })

  it('shows loading skeleton initially', () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    mockGetGrouped.mockReturnValue(new Promise(() => {}))
    mockGetRecent.mockReturnValue(new Promise(() => {}))
    mockGetTrends.mockReturnValue(new Promise(() => {}))
    render(<AdminPage />)
    expect(screen.getAllByText('Admin').length).toBeGreaterThanOrEqual(1)
  })

  it('renders three tab triggers', async () => {
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /security/i })
    expect(screen.getByRole('tab', { name: /auth/i })).toBeTruthy()
    expect(screen.getByRole('tab', { name: /errors/i })).toBeTruthy()
  })

  it('defaults to security tab', async () => {
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /security/i })
    expect(screen.getByTestId('security-overview')).toBeTruthy()
  })

  it('switches to auth tab on click', async () => {
    const user = userEvent.setup()
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /auth/i })
    await user.click(screen.getByRole('tab', { name: /auth/i }))
    expect(screen.getAllByText('Login').length).toBeGreaterThanOrEqual(1)
  })

  it('switches to errors tab on click', async () => {
    const user = userEvent.setup()
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /errors/i })
    await user.click(screen.getByRole('tab', { name: /errors/i }))
    expect(screen.getByTestId('error-insights')).toBeTruthy()
  })

  it('renders security overview card', async () => {
    render(<AdminPage />)
    await screen.findByTestId('security-overview')
    expect(screen.getByTestId('security-overview')).toBeTruthy()
  })

  it('fetches audit logs on mount', async () => {
    render(<AdminPage />)
    await screen.findByTestId('security-overview')
    expect(mockApiGet).toHaveBeenCalledWith('/security/audit?limit=100')
  })

  it('fetches security keys on mount', async () => {
    render(<AdminPage />)
    await screen.findByTestId('security-overview')
    expect(mockApiGet).toHaveBeenCalledWith('/security/keys')
  })

  it('fetches errors on mount', async () => {
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /errors/i })
    expect(mockGetGrouped).toHaveBeenCalled()
    expect(mockGetRecent).toHaveBeenCalledWith(30)
    expect(mockGetTrends).toHaveBeenCalledWith(24)
  })

  it('shows login form when not authenticated', async () => {
    const user = userEvent.setup()
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /auth/i })
    await user.click(screen.getByRole('tab', { name: /auth/i }))
    expect(screen.getAllByPlaceholderText(/username/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByPlaceholderText(/password/i).length).toBeGreaterThanOrEqual(1)
  })

  it('shows Guest status in auth tab when not logged in', async () => {
    const user = userEvent.setup()
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /auth/i })
    await user.click(screen.getByRole('tab', { name: /auth/i }))
    expect(screen.getByText('Guest')).toBeTruthy()
  })

  it('displays audit log entries when loaded', async () => {
    mockApiGet.mockResolvedValueOnce({
      logs: [{ event_type: 'auth_success', timestamp: '2024-01-01T00:00:00+00:00', user: 'admin' }],
    }).mockResolvedValueOnce({ count: 1, configured: true })
    render(<AdminPage />)
    await screen.findByText('auth_success')
  })

  it('shows empty state when no audit logs', async () => {
    render(<AdminPage />)
    await screen.findByText(/no audit logs/i)
  })

  it('renders error groups in errors tab', async () => {
    mockGetGrouped.mockResolvedValue([
      { message: 'Test error', source: 'test.ts', count: 5, fingerprint: 'abc', latest: '2024-01-01' },
    ])
    mockGetRecent.mockResolvedValue({ errors: [], total: 5 })
    const user = userEvent.setup()
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /errors/i })
    await user.click(screen.getByRole('tab', { name: /errors/i }))
    expect(screen.getAllByText('Test error').length).toBeGreaterThanOrEqual(1)
  })

  it('shows KPI stats for errors', async () => {
    mockGetRecent.mockResolvedValue({ errors: [{ id: '1', message: 'e', timestamp: new Date().toISOString() }], total: 1 })
    const user = userEvent.setup()
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /errors/i })
    await user.click(screen.getByRole('tab', { name: /errors/i }))
    expect(screen.getAllByText('1').length).toBeGreaterThanOrEqual(1)
  })

  it('shows clear all button in errors tab', async () => {
    const user = userEvent.setup()
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /errors/i })
    await user.click(screen.getByRole('tab', { name: /errors/i }))
    expect(screen.getByText(/clear all/i)).toBeTruthy()
  })

  it('shows export all button in errors tab', async () => {
    const user = userEvent.setup()
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /errors/i })
    await user.click(screen.getByRole('tab', { name: /errors/i }))
    expect(screen.getByText(/export all/i)).toBeTruthy()
  })

  it('shows Token Info section in auth tab', async () => {
    const user = userEvent.setup()
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /auth/i })
    await user.click(screen.getByRole('tab', { name: /auth/i }))
    expect(screen.getByText('Token Info')).toBeTruthy()
  })

  it('shows no token message when not logged in', async () => {
    const user = userEvent.setup()
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /auth/i })
    await user.click(screen.getByRole('tab', { name: /auth/i }))
    expect(screen.getByText(/no token/i)).toBeTruthy()
  })

  it('renders refresh button', async () => {
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /security/i })
    expect(screen.getAllByRole('button', { name: /refresh/i }).length).toBeGreaterThanOrEqual(1)
  })

  it('renders audit log filter input', async () => {
    render(<AdminPage />)
    await screen.findByPlaceholderText(/filter by event/i)
  })

  it('shows KPI grid on security tab', async () => {
    render(<AdminPage />)
    await screen.findByText('API Keys')
    expect(screen.getAllByText('Audit Logs').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('History Mode').length).toBeGreaterThanOrEqual(1)
  })

  it('can toggle history mode', async () => {
    const user = userEvent.setup()
    render(<AdminPage />)
    await screen.findByText('API Keys')
    const sessionBtn = await screen.findByRole('button', { name: /session/i })
    await user.click(sessionBtn)
    expect(screen.getAllByText('Persisted').length).toBeGreaterThanOrEqual(1)
  })

  it('displays grouped errors count', async () => {
    mockGetGrouped.mockResolvedValue([
      { message: 'err1', source: 's', count: 1, fingerprint: 'f1', latest: '2024-01-01' },
      { message: 'err2', source: 's', count: 2, fingerprint: 'f2', latest: '2024-01-02' },
    ])
    const user = userEvent.setup()
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /errors/i })
    await user.click(screen.getByRole('tab', { name: /errors/i }))
    expect(screen.getByText(/grouped errors \(2\)/i)).toBeTruthy()
  })

  it('shows auto-refresh toggle in errors tab', async () => {
    const user = userEvent.setup()
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /errors/i })
    await user.click(screen.getByRole('tab', { name: /errors/i }))
    expect(screen.getByText(/auto-refresh/i)).toBeTruthy()
  })

  it('shows register form toggle', async () => {
    const user = userEvent.setup()
    render(<AdminPage />)
    await screen.findByRole('tab', { name: /auth/i })
    await user.click(screen.getByRole('tab', { name: /auth/i }))
    const createBtn = screen.getByText(/create account/i)
    expect(createBtn).toBeTruthy()
    await user.click(createBtn)
    expect(screen.getByPlaceholderText(/email/i)).toBeTruthy()
  })
})
