// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import { TrainingHistoryView } from './TrainingHistoryView'
import { trainingJobsController } from '@/lib/training-controller'

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    list: vi.fn(),
    recover: vi.fn(),
  },
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
}))

vi.mock('@sloughgpt/strui', async () => {
  const actual = await vi.importActual('@sloughgpt/strui')
  return {
    ...actual,
    Skeleton: ({ className }: any) => <div data-testid="skeleton" className={className} />,
    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => (
      <div data-testid="action-card">
        <h3>{title}</h3>
        {children}
      </div>
    ),
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => (
      <textarea value={value} onChange={onChange} {...props} />
    ),
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
  }
})

const JOBS = [
  {
    id: '1',
    name: 'Job 1',
    status: 'completed',
    progress: 100,
    created_at: '2026-01-15T10:00:00Z',
    method: 'distill',
    loss: 0.5,
    epochs: 10,
    checkpoint: 'cp1',
  },
  {
    id: '2',
    name: 'Job 2',
    status: 'failed',
    progress: 50,
    created_at: '2026-01-14T10:00:00Z',
    method: 'native',
    error: 'OOM',
  },
] as any[]

describe('TrainingHistoryView', () => {
  const mockList = vi.mocked(trainingJobsController.list)
  const mockRecover = vi.mocked(trainingJobsController.recover)
  const mockToast = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue(JOBS)
  })

  it('shows loading state', () => {
    mockList.mockImplementation(() => new Promise(() => {}))
    render(<TrainingHistoryView addToast={mockToast} />)
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(0)
  })

  it('shows empty state when no jobs', async () => {
    mockList.mockResolvedValue([])
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('No training jobs yet. Start training to see history.')).toBeTruthy()
    })
  })

  it('renders job list', async () => {
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 1').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Job 2').length).toBeGreaterThan(0)
    })
  })

  it('renders jobs without created_at (backend strips null fields)', async () => {
    mockList.mockResolvedValue([
      { id: '9', name: 'NoDateJob', status: 'running', progress: 10 } as any,
    ])
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('NoDateJob').length).toBeGreaterThan(0)
    })
  })

  it('shows status filter buttons', async () => {
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('All (2)')).toBeTruthy()
    })
  })

  it('expands job details on click', async () => {
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 1').length).toBeGreaterThan(0)
    })
    fireEvent.click(screen.getAllByText('Job 1')[0])
    await waitFor(() => {
      expect(screen.getByText('Loss: 0.5000')).toBeTruthy()
      expect(screen.getByText('Epochs: 10')).toBeTruthy()
      expect(screen.getByText('Checkpoint: cp1')).toBeTruthy()
    })
  })

  it('shows error on fetch failure', async () => {
    mockList.mockRejectedValue(new Error('Network error'))
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith('Could not fetch training history', 'error')
    })
  })

  it('exports training history as JSON', async () => {
    const { downloadJson } = await import('@/lib/download-utils')
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 1').length).toBeGreaterThan(0)
    })
    fireEvent.click(screen.getAllByText('Export')[0])
    await waitFor(() => {
      expect(downloadJson).toHaveBeenCalled()
      expect(mockToast).toHaveBeenCalledWith('Training history exported', 'success')
    })
  })

  it('paginates jobs at 20 per page', async () => {
    const manyJobs = Array.from({ length: 25 }, (_, i) => ({
      id: `${i}`,
      name: `Job ${i}`,
      status: 'completed',
      progress: 100,
      created_at: `2026-01-${String(15 - (i % 15)).padStart(2, '0')}T10:00:00Z`,
    }))
    mockList.mockResolvedValue(manyJobs as any[])
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 0').length).toBeGreaterThan(0)
    })
    expect(screen.queryByText('Job 20')).toBeNull()
    expect(screen.getByText('1–20 of 25')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Next' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    await waitFor(() => {
      expect(screen.getAllByText('Job 20').length).toBeGreaterThan(0)
    })
    expect(screen.queryByText('Job 0')).toBeNull()
    expect(screen.getByText('21–25 of 25')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Prev' }))
    await waitFor(() => {
      expect(screen.getAllByText('Job 0').length).toBeGreaterThan(0)
    })
  })

  it('does not show pagination when 20 or fewer jobs', async () => {
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 1').length).toBeGreaterThan(0)
    })
    expect(screen.queryByText('1–2 of 2')).toBeNull()
  })

  it('shows Resume only for interrupted/failed jobs', async () => {
    mockList.mockResolvedValue([JOBS[0], { ...JOBS[1], status: 'interrupted' }] as any[])
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 2').length).toBeGreaterThan(0)
    })
    const resumeButtons = screen.getAllByRole('button', { name: 'Resume' })
    expect(resumeButtons).toHaveLength(1)
  })

  it('does not show Resume for completed/running jobs', async () => {
    mockList.mockResolvedValue([JOBS[0], { ...JOBS[1], status: 'running' }] as any[])
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 2').length).toBeGreaterThan(0)
    })
    expect(screen.queryByRole('button', { name: 'Resume' })).toBeNull()
  })

  it('resumes a failed job and fires onRecovered', async () => {
    mockRecover.mockResolvedValue({
      status: 'recovered',
      recovery_job_id: 'recovery_1',
      message: 'Resumed from checkpoint',
    })
    const onRecovered = vi.fn()
    render(<TrainingHistoryView addToast={mockToast} onRecovered={onRecovered} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 2').length).toBeGreaterThan(0)
    })
    await act(async () => {
      screen.getByRole('button', { name: 'Resume' }).click()
    })
    await waitFor(() => {
      expect(mockRecover).toHaveBeenCalledWith('2')
      expect(mockToast).toHaveBeenCalledWith('Resumed from checkpoint', 'success')
      expect(onRecovered).toHaveBeenCalledWith(
        expect.objectContaining({ recovery_job_id: 'recovery_1' }),
      )
    })
    expect(mockList).toHaveBeenCalledTimes(2)
  })

  it('shows error toast when resume fails', async () => {
    mockRecover.mockRejectedValue(new Error('no checkpoint'))
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 2').length).toBeGreaterThan(0)
    })
    await act(async () => {
      screen.getByRole('button', { name: 'Resume' }).click()
    })
    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith('Could not resume job: no checkpoint', 'error')
    })
  })

  it('disables Resume while a resume is in flight', async () => {
    mockRecover.mockImplementation(() => new Promise(() => {}))
    render(<TrainingHistoryView addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Job 2').length).toBeGreaterThan(0)
    })
    const btn = screen.getByRole('button', { name: 'Resume' })
    await act(async () => {
      btn.click()
    })
    expect(screen.getByRole('button', { name: 'Resuming…' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Resuming…' })).toBeDisabled()
  })
})
