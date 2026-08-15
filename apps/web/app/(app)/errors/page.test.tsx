import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => {
    const C = () => <span data-testid={'icon-' + name}>{name}</span>
    C.displayName = 'Icon' + name
    return C
  }
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Card: passthrough,
    CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    CardContent: passthrough,
    Button: ({ children, onClick, disabled, variant, className, 'aria-label': ariaLabel }: any) => (
      <button onClick={onClick} disabled={disabled} className={className} data-variant={variant} aria-label={ariaLabel}>{children}</button>
    ),
    Input: ({ value, onChange, placeholder, className, type }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} className={className} type={type} />
    ),
    IconRefresh: iconMock('refresh'),
    IconTrash: iconMock('trash'),
    IconDownload: iconMock('download'),
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    StatCard: ({ label, value }: any) => <div>{label}: {value}</div>,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    EmptyCard: ({ title, description }: any) => <div data-testid="empty-card"><div>{title}</div><div>{description}</div></div>,
  }
})

const { mockGetGrouped, mockGetRecent, mockGetTrends, mockClear, mockExport, mockDownloadJson, mockAddToast } = vi.hoisted(() => ({
  mockGetGrouped: vi.fn(),
  mockGetRecent: vi.fn(),
  mockGetTrends: vi.fn(),
  mockClear: vi.fn(),
  mockExport: vi.fn(),
  mockDownloadJson: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/errors-controller', () => ({
  errorsController: {
    getGrouped: mockGetGrouped,
    getRecent: mockGetRecent,
    getTrends: mockGetTrends,
    clear: mockClear,
    export: mockExport,
  },
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))
vi.mock('@/lib/download-utils', () => ({
  downloadJson: mockDownloadJson,
}))
vi.mock('@/components/errors/ErrorInsightsCard', () => ({
  ErrorInsightsCard: ({ grouped, recent }: any) => (grouped.length > 0 || recent.length > 0 ? <div data-testid="error-insights-card" /> : null),
}))

import ErrorsPage from './page'

afterEach(() => { cleanup() })
beforeEach(() => {
  vi.clearAllMocks()
  mockGetGrouped.mockResolvedValue([])
  mockGetRecent.mockResolvedValue({ errors: [], total: 0 })
  mockGetTrends.mockResolvedValue([])
})

async function renderLoaded() {
  render(<ErrorsPage />)
  await waitFor(() => { expect(screen.getByText('0 total errors')).toBeTruthy() })
}

describe('ErrorsPage', () => {
  it('shows loading skeleton and fetches all data on mount', () => {
    mockGetGrouped.mockReturnValue(new Promise(() => {}))
    render(<ErrorsPage />)
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(1)
    expect(mockGetGrouped).toHaveBeenCalled()
    expect(mockGetRecent).toHaveBeenCalledWith(30)
    expect(mockGetTrends).toHaveBeenCalledWith(24)
  })

  it('displays grouped, recent, and trend data after load', async () => {
    const latest = '2026-08-01T10:00:00Z'
    mockGetGrouped.mockResolvedValue([{
      fingerprint: 'f1',
      message: 'TypeError: x is not a function',
      source: 'app.ts',
      count: 5,
      latest,
      sample_id: 's1',
      sample_url: 'https://example.com/app.js',
      sample_line: 42,
    }])
    mockGetRecent.mockResolvedValue({
      errors: [{ id: 'e1', message: 'Failed to fetch /api/x', source: 'api.ts', timestamp: latest, fingerprint: 'f2' }],
      total: 7,
    })
    mockGetTrends.mockResolvedValue([
      { hour: '2026-08-01T10:00:00Z', count: 3 },
      { hour: '2026-08-01T11:00:00Z', count: 5 },
    ])
    render(<ErrorsPage />)
    await waitFor(() => { expect(screen.getByText('7 total errors')).toBeTruthy() })
    expect(screen.getByText('Grouped Errors (1)')).toBeTruthy()
    expect(screen.getByText('TypeError: x is not a function')).toBeTruthy()
    expect(screen.getByText(/app\.ts/)).toBeTruthy()
    expect(screen.getByText('https://example.com/app.js')).toBeTruthy()
    expect(screen.getByText(/:42/)).toBeTruthy()
    expect(screen.getByText('×5')).toBeTruthy()
    expect(screen.getByText(new Date(latest).toLocaleDateString())).toBeTruthy()
    expect(screen.getByText('Recent Errors')).toBeTruthy()
    expect(screen.getByText('Failed to fetch /api/x')).toBeTruthy()
    expect(screen.getByText(new Date(latest).toLocaleTimeString())).toBeTruthy()
    expect(screen.getByText('Hourly Trend (24h)')).toBeTruthy()
    expect(screen.getByText('10:00:00Z')).toBeTruthy()
    expect(screen.getByText('11:00:00Z')).toBeTruthy()
    expect(screen.getByTestId('error-insights-card')).toBeTruthy()
  })

  it('shows empty states when there are no errors', async () => {
    await renderLoaded()
    expect(screen.getByText('No errors logged.')).toBeTruthy()
    expect(screen.getByText('No recent errors.')).toBeTruthy()
    expect(screen.queryByTestId('error-insights-card')).toBeFalsy()
  })

  it('shows error toast when initial fetch fails', async () => {
    mockGetGrouped.mockRejectedValueOnce(new Error('boom'))
    render(<ErrorsPage />)
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Failed to load error data', 'error') })
  })

  it('filters grouped errors by search term', async () => {
    mockGetGrouped.mockResolvedValue([
      { fingerprint: 'f1', message: 'TypeError: x is not a function', source: 'app.ts', count: 1, latest: '', sample_id: '', sample_url: '', sample_line: null },
      { fingerprint: 'f2', message: 'Failed to load model', source: 'model.ts', count: 2, latest: '', sample_id: '', sample_url: '', sample_line: null },
    ])
    await renderLoaded()
    expect(screen.getByText('TypeError: x is not a function')).toBeTruthy()
    expect(screen.getByText('Failed to load model')).toBeTruthy()
    const searchInput = screen.getByPlaceholderText('Search errors...')
    await act(async () => { fireEvent.change(searchInput, { target: { value: 'typeerror' } }) })
    expect(screen.getByText('TypeError: x is not a function')).toBeTruthy()
    expect(screen.queryByText('Failed to load model')).toBeFalsy()
  })

  it('clears errors and refreshes data', async () => {
    mockClear.mockResolvedValue(undefined)
    await renderLoaded()
    expect(mockGetGrouped).toHaveBeenCalledTimes(1)
    await act(async () => { screen.getByText('Clear All').click() })
    await waitFor(() => { expect(mockClear).toHaveBeenCalled() })
    expect(mockGetGrouped).toHaveBeenCalledTimes(2)
  })

  it('shows Clearing... and disables button while clearing', async () => {
    let resolveClear!: (v: void) => void
    mockClear.mockReturnValue(new Promise(r => { resolveClear = r }))
    await renderLoaded()
    await act(async () => { screen.getByText('Clear All').click() })
    await waitFor(() => { expect(screen.getByText('Clearing...')).toBeTruthy() })
    const clearingBtn = screen.getByText('Clearing...').closest('button') as HTMLButtonElement
    expect(clearingBtn.disabled).toBe(true)
    await act(async () => { resolveClear() })
    await waitFor(() => { expect(screen.getByText('Clear All')).toBeTruthy() })
  })

  it('shows error toast when clearing fails', async () => {
    mockClear.mockRejectedValue(new Error('boom'))
    await renderLoaded()
    await act(async () => { screen.getByText('Clear All').click() })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Failed to clear errors', 'error') })
  })

  it('exports errors as JSON via downloadJson', async () => {
    mockExport.mockResolvedValue({ errors: [] })
    await renderLoaded()
    await act(async () => { screen.getByText('Export All').click() })
    await waitFor(() => { expect(mockExport).toHaveBeenCalled() })
    expect(mockDownloadJson).toHaveBeenCalledTimes(1)
    const filename = mockDownloadJson.mock.calls[0][1]
    expect(String(filename)).toMatch(/^errors-\d+\.json$/)
  })

  it('shows error toast when export fails', async () => {
    mockExport.mockRejectedValue(new Error('boom'))
    await renderLoaded()
    await act(async () => { screen.getByText('Export All').click() })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Failed to export errors', 'error') })
  })

  it('toggles auto-refresh label', async () => {
    await renderLoaded()
    await act(async () => { screen.getByText('Auto-refresh').click() })
    expect(screen.getByText('Auto-refresh ON')).toBeTruthy()
  })

  it('re-fetches data on the interval when auto-refresh is on', async () => {
    vi.useFakeTimers()
    render(<ErrorsPage />)
    await act(async () => {})
    expect(mockGetGrouped).toHaveBeenCalledTimes(1)
    await act(async () => { screen.getByText('Auto-refresh').click() })
    expect(screen.getByText('Auto-refresh ON')).toBeTruthy()
    await act(async () => { vi.advanceTimersByTime(10000) })
    await act(async () => {})
    expect(mockGetGrouped).toHaveBeenCalledTimes(2)
    cleanup()
    vi.useRealTimers()
  })

  it('re-fetches data when refresh button is clicked', async () => {
    await renderLoaded()
    expect(mockGetGrouped).toHaveBeenCalledTimes(1)
    const refreshBtn = screen.getByTestId('icon-refresh').closest('button')
    await act(async () => { refreshBtn!.click() })
    await waitFor(() => { expect(mockGetGrouped).toHaveBeenCalledTimes(2) })
  })
})
