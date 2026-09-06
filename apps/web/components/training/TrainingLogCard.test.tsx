// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import { TrainingLogCard } from './TrainingLogCard'
import { trainingJobsController } from '@/lib/training-controller'

vi.mock('@sloughgpt/strui', () => ({
  ActionCard: ({ children, title, actions, className }: any) => (
    <div className={className}>
      <div data-testid="card-title">{title}</div>
      {actions && <div>{actions}</div>}
      {children}
    </div>
  ),
  Button: ({ children, onClick, disabled, className }: any) => (
    <button onClick={onClick} disabled={disabled} className={className}>{children}</button>
  ),
  Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    getTrainingLog: vi.fn(),
  },
}))

describe('TrainingLogCard', () => {
  const mockGetTrainingLog = vi.mocked(trainingJobsController.getTrainingLog)

  beforeEach(() => {
    vi.clearAllMocks()
    vi.restoreAllMocks()
    mockGetTrainingLog.mockResolvedValue(['line 1', 'line 2', 'line 3'])
  })

  it('renders collapsed by default', () => {
    const { unmount } = render(<TrainingLogCard trainingRunning={false} />)
    expect(screen.getByText('Training logs')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Show' })).toBeTruthy()
    expect(screen.queryByText('line 1')).toBeNull()
    unmount()
  })

  it('expands and fetches logs on Show click', async () => {
    const { unmount } = render(<TrainingLogCard trainingRunning={false} />)
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Show' }))
    })
    await waitFor(() => {
      expect(mockGetTrainingLog).toHaveBeenCalled()
    })
    expect(screen.getByText('line 1')).toBeTruthy()
    expect(screen.getByText('line 2')).toBeTruthy()
    expect(screen.getByText('line 3')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Hide' })).toBeTruthy()
    unmount()
  })

  it('collapses on Hide click', async () => {
    const { unmount } = render(<TrainingLogCard trainingRunning={false} />)
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Show' }))
    })
    await waitFor(() => {
      expect(screen.getByText('line 1')).toBeTruthy()
    })
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Hide' }))
    })
    expect(screen.queryByText('line 1')).toBeNull()
    expect(screen.getByRole('button', { name: 'Show' })).toBeTruthy()
    unmount()
  })

  it('shows empty message when no logs', async () => {
    mockGetTrainingLog.mockResolvedValue([])
    const { unmount } = render(<TrainingLogCard trainingRunning={false} />)
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Show' }))
    })
    await waitFor(() => {
      expect(screen.getByText('No logs yet.')).toBeTruthy()
    })
    unmount()
  })

  it('shows loading skeleton while fetching', async () => {
    let resolveFn: (value: string[]) => void
    mockGetTrainingLog.mockImplementation(() => new Promise(r => { resolveFn = r }))
    const { unmount } = render(<TrainingLogCard trainingRunning={false} />)
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Show' }))
    })
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(0)
    await act(async () => { resolveFn!([]) })
    expect(screen.queryByTestId('skeleton')).toBeNull()
    unmount()
  })

  it('displays live indicator when training running and expanded', async () => {
    const { unmount } = render(<TrainingLogCard trainingRunning={true} />)
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Show' }))
    })
    await waitFor(() => {
      expect(screen.getByText('live')).toBeTruthy()
    })
    unmount()
  })

  it('does not display live indicator when training not running', async () => {
    const { unmount } = render(<TrainingLogCard trainingRunning={false} />)
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Show' }))
    })
    await waitFor(() => {
      expect(mockGetTrainingLog).toHaveBeenCalled()
    })
    expect(screen.queryByText('live')).toBeNull()
    unmount()
  })

  it('caps visible log lines at 500 and shows overflow message', async () => {
    const manyLines = Array.from({ length: 600 }, (_, i) => `line ${i}`)
    mockGetTrainingLog.mockResolvedValue(manyLines)
    const { unmount } = render(<TrainingLogCard trainingRunning={false} />)
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Show' }))
    })
    await waitFor(() => {
      expect(screen.getByText('Showing last 500 of 600 lines')).toBeTruthy()
    })
    expect(screen.queryByText('line 0')).toBeNull()
    expect(screen.queryByText('line 99')).toBeNull()
    expect(screen.queryByText('line 100')).toBeTruthy()
    expect(screen.queryByText('line 599')).toBeTruthy()
    unmount()
  })
})
