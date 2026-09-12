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

    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
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
