import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'

const mockErrors = vi.fn()
const mockTotalCount = vi.fn()
const mockDismiss = vi.fn()
const mockClear = vi.fn()

vi.mock('@/lib/error-store', () => ({
  useErrorStore: (sel: (s: any) => any) => sel({
    errors: mockErrors(),
    totalErrorCount: mockTotalCount(),
    dismissError: mockDismiss,
    clearErrors: mockClear,
  }),
}))

afterEach(cleanup)

describe('ActivityTicker', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockTotalCount.mockReturnValue(0)
  })

  it('shows no-error state', async () => {
    mockErrors.mockReturnValue([])
    const { ActivityTicker } = await import('@/components/ActivityTicker')
    render(<ActivityTicker />)
    expect(screen.getAllByText('No errors').length).toBeGreaterThanOrEqual(1)
  })

  it('shows error count badge', async () => {
    mockErrors.mockReturnValue([
      { id: '1', title: 'Test', message: 'msg', severity: 'error', timestamp: Date.now(), count: 1 },
    ])
    mockTotalCount.mockReturnValue(1)
    const { ActivityTicker } = await import('@/components/ActivityTicker')
    render(<ActivityTicker />)
    expect(screen.getAllByText('1').length).toBeGreaterThanOrEqual(1)
  })

  it('shows error title when not "Error"', async () => {
    mockErrors.mockReturnValue([
      { id: '1', title: 'Network Error', message: 'Failed to fetch', severity: 'error', timestamp: Date.now(), count: 1 },
    ])
    mockTotalCount.mockReturnValue(1)
    const { ActivityTicker } = await import('@/components/ActivityTicker')
    render(<ActivityTicker />)
    expect(screen.getAllByText('Network Error').length).toBeGreaterThanOrEqual(1)
  })

  it('shows truncated message when title is "Error"', async () => {
    mockErrors.mockReturnValue([
      { id: '1', title: 'Error', message: 'Something went wrong here', severity: 'error', timestamp: Date.now(), count: 1 },
    ])
    mockTotalCount.mockReturnValue(1)
    const { ActivityTicker } = await import('@/components/ActivityTicker')
    render(<ActivityTicker />)
    expect(screen.getAllByText('Something went wrong here').length).toBeGreaterThanOrEqual(1)
  })

  it('shows time ago', async () => {
    mockErrors.mockReturnValue([
      { id: '1', title: 'E', message: 'm', severity: 'error', timestamp: Date.now(), count: 1 },
    ])
    mockTotalCount.mockReturnValue(1)
    const { ActivityTicker } = await import('@/components/ActivityTicker')
    render(<ActivityTicker />)
    expect(screen.getAllByText('just now').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onExpand when clicked', async () => {
    const onExpand = vi.fn()
    mockErrors.mockReturnValue([
      { id: '1', title: 'E', message: 'm', severity: 'error', timestamp: Date.now(), count: 1 },
    ])
    mockTotalCount.mockReturnValue(1)
    const { ActivityTicker } = await import('@/components/ActivityTicker')
    render(<ActivityTicker onExpand={onExpand} />)
    fireEvent.click(screen.getAllByRole('button')[0])
    expect(onExpand).toHaveBeenCalled()
  })
})

describe('ErrorList', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('returns null when empty', async () => {
    mockErrors.mockReturnValue([])
    const { ErrorList } = await import('@/components/ActivityTicker')
    const { container } = render(<ErrorList />)
    expect(container.innerHTML).toBe('')
  })

  it('renders error items', async () => {
    mockErrors.mockReturnValue([
      { id: '1', title: 'Test Error', message: 'Details', severity: 'error', timestamp: Date.now(), count: 1 },
    ])
    const { ErrorList } = await import('@/components/ActivityTicker')
    render(<ErrorList />)
    expect(screen.getAllByText('Test Error').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Details').length).toBeGreaterThanOrEqual(1)
  })

  it('shows unique error count', async () => {
    mockErrors.mockReturnValue([
      { id: '1', title: 'A', message: '', severity: 'error', timestamp: Date.now(), count: 1 },
      { id: '2', title: 'B', message: '', severity: 'error', timestamp: Date.now(), count: 1 },
    ])
    const { ErrorList } = await import('@/components/ActivityTicker')
    render(<ErrorList />)
    expect(screen.getAllByText('2 unique errors').length).toBeGreaterThanOrEqual(1)
  })

  it('shows repeat count for deduped errors', async () => {
    mockErrors.mockReturnValue([
      { id: '1', title: 'A', message: '', severity: 'error', timestamp: Date.now(), count: 3 },
    ])
    const { ErrorList } = await import('@/components/ActivityTicker')
    render(<ErrorList />)
    expect(screen.getAllByText('×3').length).toBeGreaterThanOrEqual(1)
  })

  it('dismisses error', async () => {
    mockErrors.mockReturnValue([
      { id: 'err-x', title: 'E', message: '', severity: 'error', timestamp: Date.now(), count: 1 },
    ])
    const { ErrorList } = await import('@/components/ActivityTicker')
    render(<ErrorList />)
    fireEvent.click(screen.getAllByLabelText('Dismiss')[0])
    expect(mockDismiss).toHaveBeenCalledWith('err-x')
  })

  it('clears all errors', async () => {
    mockErrors.mockReturnValue([
      { id: '1', title: 'E', message: '', severity: 'error', timestamp: Date.now(), count: 1 },
    ])
    const { ErrorList } = await import('@/components/ActivityTicker')
    render(<ErrorList />)
    fireEvent.click(screen.getAllByText('Clear all')[0])
    expect(mockClear).toHaveBeenCalled()
  })

  it('renders source when present', async () => {
    mockErrors.mockReturnValue([
      { id: '1', title: 'E', message: '', severity: 'error', timestamp: Date.now(), count: 1, source: 'inference' },
    ])
    const { ErrorList } = await import('@/components/ActivityTicker')
    render(<ErrorList />)
    expect(screen.getAllByText(/inference/).length).toBeGreaterThanOrEqual(1)
  })
})
