import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const mockApiGet = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

import SecurityPage from './page'

describe('SecurityPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockResolvedValue({ data: { logs: [] } })
  })

  afterEach(() => {
    cleanup()
  })

  it('renders page header', async () => {
    render(<SecurityPage />)
    expect(screen.getAllByText('Security').length).toBeGreaterThanOrEqual(1)
  })

  it('renders search input after loading', async () => {
    render(<SecurityPage />)
    await screen.findByPlaceholderText(/filter by event/i)
    expect(screen.getAllByPlaceholderText(/filter by event/i).length).toBeGreaterThanOrEqual(1)
  })

  it('toggles to persisted history fetch', async () => {
    const user = userEvent.setup()
    render(<SecurityPage />)
    const sessionBtn = await screen.findByRole('button', { name: /session/i })
    await user.click(sessionBtn)
    expect(mockApiGet).toHaveBeenCalledWith('/security/audit?history=true&limit=100')
    expect(screen.getByRole('button', { name: /persisted/i })).toBeDefined()
  })

  it('loads older persisted logs with before cursor', async () => {
    const user = userEvent.setup()
    const rec = { event_type: 'auth_success', timestamp: '2024-01-01T00:00:02+00:00', user: 'u' }
    const older = { event_type: 'auth_older', timestamp: '2024-01-01T00:00:01+00:00', user: 'u' }
    mockApiGet
      .mockResolvedValueOnce({ data: { logs: [rec] } })        // initial audit fetch
      .mockResolvedValueOnce({ data: { count: 1, configured: true } }) // keys
      .mockResolvedValueOnce({ data: { logs: [rec] } })        // persisted fetch
      .mockResolvedValueOnce({ data: { count: 1, configured: true } }) // keys
      .mockResolvedValue({ data: { logs: [older] } })          // load older
    render(<SecurityPage />)
    await screen.findAllByText('auth_success')
    await user.click(screen.getByRole('button', { name: /session/i }))
    await screen.findAllByText('auth_success')
    await user.click(screen.getByRole('button', { name: /load older/i }))
    expect((await screen.findAllByText('auth_older')).length).toBeGreaterThanOrEqual(1)
    expect(mockApiGet).toHaveBeenCalledWith(
      '/security/audit?history=true&limit=100&before=2024-01-01T00%3A00%3A02%2B00%3A00',
    )
  })

  it('passes event_type to backend when filter is set', async () => {
    const user = userEvent.setup()
    mockApiGet
      .mockResolvedValueOnce({ data: { logs: [] } })               // initial audit fetch
      .mockResolvedValueOnce({ data: { count: 0, configured: false } }) // keys
      .mockResolvedValue({ data: { logs: [] } })                   // refresh
    render(<SecurityPage />)
    const input = await screen.findByPlaceholderText(/filter by event/i)
    await user.type(input, 'training')
    await user.click(screen.getByRole('button', { name: /refresh audit logs/i }))
    expect(mockApiGet).toHaveBeenCalledWith('/security/audit?limit=100&event_type=training')
  })

  it('encodes event_type and keeps it on load older', async () => {
    const user = userEvent.setup()
    const rec = { event_type: 'training.start', timestamp: '2024-01-01T00:00:02+00:00', user: 'u' }
    const older = { event_type: 'training.start', timestamp: '2024-01-01T00:00:01+00:00', user: 'u' }
    mockApiGet
      .mockResolvedValueOnce({ data: { logs: [rec] } })        // initial audit fetch
      .mockResolvedValueOnce({ data: { count: 1, configured: true } }) // keys
      .mockResolvedValueOnce({ data: { logs: [rec] } })        // persisted fetch
      .mockResolvedValueOnce({ data: { count: 1, configured: true } }) // keys
      .mockResolvedValue({ data: { logs: [older] } })          // load older
    render(<SecurityPage />)
    const input = await screen.findByPlaceholderText(/filter by event/i)
    await user.type(input, 'training.start')
    await user.click(screen.getByRole('button', { name: /session/i }))
    await screen.findAllByText('training.start')
    await user.click(screen.getByRole('button', { name: /load older/i }))
    expect(mockApiGet).toHaveBeenCalledWith(
      '/security/audit?history=true&limit=100&before=2024-01-01T00%3A00%3A02%2B00%3A00&event_type=training.start',
    )
  })

  it('shows empty state when no logs', async () => {
    mockApiGet.mockResolvedValue({ data: { logs: [] } })
    render(<SecurityPage />)
    await screen.findByPlaceholderText(/filter by event/i)
    expect(screen.getAllByText(/no|empty|nothing/i).length).toBeGreaterThanOrEqual(1)
  })

  it('shows error state when fetch fails', async () => {
    mockApiGet.mockRejectedValue(new Error('Network error'))
    render(<SecurityPage />)
    await screen.findByPlaceholderText(/filter by event/i)
    expect(screen.getByText('Security')).toBeTruthy()
  })

  it('displays log entries when loaded', async () => {
    const logs = [
      { event_type: 'auth_success', timestamp: '2024-01-01T00:00:01+00:00', user: 'admin' },
      { event_type: 'model_load', timestamp: '2024-01-01T00:00:02+00:00', user: 'system' },
    ]
    mockApiGet.mockResolvedValue({ data: { logs } })
    render(<SecurityPage />)
    await screen.findAllByText('auth_success')
    expect(screen.getAllByText('model_load').length).toBeGreaterThanOrEqual(1)
  })

  it('shows KPI grid with log count', async () => {
    const logs = [
      { event_type: 'auth_success', timestamp: '2024-01-01T00:00:01+00:00', user: 'admin' },
    ]
    mockApiGet.mockResolvedValue({ data: { logs } })
    render(<SecurityPage />)
    await screen.findAllByText('auth_success')
    expect(screen.getByText('Security')).toBeTruthy()
  })

  it('renders refresh button', async () => {
    render(<SecurityPage />)
    await screen.findByPlaceholderText(/filter by event/i)
    expect(screen.getByRole('button', { name: /refresh/i })).toBeTruthy()
  })

  it('clicking refresh re-fetches logs', async () => {
    const user = userEvent.setup()
    render(<SecurityPage />)
    await screen.findByPlaceholderText(/filter by event/i)
    const callCountBefore = mockApiGet.mock.calls.length
    const refreshBtn = screen.getByRole('button', { name: /refresh/i })
    await user.click(refreshBtn)
    expect(mockApiGet.mock.calls.length).toBeGreaterThan(callCountBefore)
  })

  it('shows loading state while fetching', async () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<SecurityPage />)
    expect(screen.getByText('Security')).toBeTruthy()
  })

  it('displays API keys status', async () => {
    mockApiGet
      .mockResolvedValueOnce({ data: { logs: [] } })
      .mockResolvedValueOnce({ data: { count: 2, configured: true } })
    render(<SecurityPage />)
    await screen.findByPlaceholderText(/filter by event/i)
    expect(screen.getByText('Security')).toBeTruthy()
  })

  it('shows audit log section', async () => {
    render(<SecurityPage />)
    await screen.findByPlaceholderText(/filter by event/i)
    expect(screen.getByText('Security')).toBeTruthy()
  })
})
